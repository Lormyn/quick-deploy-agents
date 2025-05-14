#  Copyright (C) 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# --- Standard Library Imports ---
import asyncio
import importlib
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import vertexai
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions
from nicegui import Client, ui
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

# --- Google Cloud & Auth Imports ---
try:
    import google.auth
    import google.auth.transport.requests
    import requests
    from google.cloud import resourcemanager_v3
except ImportError as e:
    print(f"Error: Could not import Google API libraries. {e}")
    print("Please install them: pip install requests google-auth google-cloud-resource-manager")
    sys.exit(1)

# --- Configuration Loading ---
try:
    from deployment_utils.agentspace_lister import (
        get_agentspace_apps_from_projectid,  # Used for Register & Deregister
    )
    from deployment_utils.constants import (
        SUPPORTED_REGIONS,
        WEBUI_AGENTDEPLOYMENT_HELPTEXT,
    )  # Import the help text
    from deployment_utils.deployment_configs import (
        AGENT_CONFIGS,  # Used for Deploy & Register
    )
except ImportError as e:
    print(
        "Error: Could not import from 'deployment_utils'. "
        f"Ensure 'deployment_configs.py' and 'constants.py' exist. Details: {e}"
    )
    AGENT_CONFIGS = {"error": {"ae_display_name": "Import Error"}}
    SUPPORTED_REGIONS = ["us-central1"]
    WEBUI_AGENTDEPLOYMENT_HELPTEXT = "Error: Help text constant not found." # Fallback
    get_agentspace_apps_from_projectid = None # Indicate function is missing
    IMPORT_ERROR_MESSAGE = (
        "Failed to import 'AGENT_CONFIGS', 'SUPPORTED_REGIONS', 'WEBUI_AGENTDEPLOYMENT_HELPTEXT', or 'get_agentspace_apps_from_projectid' from 'deployment_utils'. "
        "Please ensure 'deployment_configs.py', 'constants.py', and 'agentspace_lister.py' exist in the 'deployment_utils' directory "
        "relative to this script, and that the directory contains an `__init__.py` file. Run: pip install -r requirements.txt"
    )
else:
    IMPORT_ERROR_MESSAGE = None

# --- Constants ---
_BASE_REQUIREMENTS = [
    "google-adk (>=0.3.0)",
    "google-cloud-aiplatform[adk, agent_engines]",
    "python-dotenv",
    "requests",
    "google-cloud-resource-manager",
]

# --- Helper Functions ---

def init_vertex_ai(project_id: str, location: str, staging_bucket: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Initializes Vertex AI SDK. Staging bucket is optional.
    Returns:
        Tuple[bool, Optional[str]]: (success_status, error_message_or_none)
    """
    try:
        bucket_info = f"(Bucket: gs://{staging_bucket})" if staging_bucket else "(No bucket specified)"
        print(f"Initializing Vertex AI SDK for {project_id}/{location} {bucket_info}...")
        init_kwargs = {"project": project_id, "location": location}
        if staging_bucket:
            init_kwargs["staging_bucket"] = f"gs://{staging_bucket}"
        vertexai.init(**init_kwargs)
        print("Vertex AI initialized successfully.")
        return True, None
    except google_exceptions.NotFound:
        bucket_error = f"or Bucket 'gs://{staging_bucket}' invalid/inaccessible" if staging_bucket else ""
        msg = f"Error: Project '{project_id}' or Location '{location}' not found, or Vertex AI API not enabled, {bucket_error}."
        print(msg)
        return False, msg
    except Exception as e:
        msg = f"Error initializing Vertex AI SDK: {e}"
        print(msg)
        return False, msg

def get_project_number_sync(project_id: str) -> Optional[str]:
    """Gets the GCP project number from the project ID (Synchronous version)."""
    try:
        client = resourcemanager_v3.ProjectsClient()
        request = resourcemanager_v3.GetProjectRequest(name=f"projects/{project_id}")
        project = client.get_project(request=request)
        return project.name.split('/')[-1]
    except Exception as e:
        print(f"Error getting project number for '{project_id}': {e}")
        return None

async def get_project_number(project_id: str) -> Optional[str]:
    """Gets the GCP project number from the project ID (Async wrapper)."""
    if not project_id: return None
    return await asyncio.to_thread(get_project_number_sync, project_id)


async def get_agent_root_nicegui(agent_config: dict) -> Tuple[Optional[Any], Optional[str]]:
    """
    Dynamically imports the root_agent for deployment.
    """
    module_path = agent_config.get("module_path")
    var_name = agent_config.get("root_variable")

    if not module_path or not var_name:
        return None, (
            "Agent configuration is missing 'module_path' or 'root_variable'.\n"
            f"Config provided: {agent_config}"
        )

    try:
        print(f"Importing '{var_name}' from module '{module_path}'...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path: sys.path.insert(0, script_dir)
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

        agent_module = importlib.import_module(module_path)
        root_agent = getattr(agent_module, var_name)
        print("Successfully imported root agent.")
        return root_agent, None
    except ImportError:
        tb_str = traceback.format_exc()
        return None, (
            f"Failed to import module '{module_path}'.\n"
            "Check 'module_path' in deployment_configs.py and ensure the module exists.\n\n"
            f"Traceback:\n{tb_str}"
        )
    except AttributeError:
        return None, (
            f"Module '{module_path}' does not have an attribute named '{var_name}'.\n"
            "Check 'root_variable' in deployment_configs.py."
        )
    except Exception as e:
        return None, f"An unexpected error occurred during agent import: {e}"

async def update_timer(start_time: float, timer_label: ui.label, stop_event: asyncio.Event, status_area: ui.element):
    """Updates the timer label every second until stop_event is set."""
    while not stop_event.is_set():
        elapsed_seconds = time.monotonic() - start_time
        minutes, seconds = divmod(int(elapsed_seconds), 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        try:
            with status_area: # Use the specific status area passed
                timer_label.set_text(f"Elapsed Time: {time_str}")
        except Exception as e:
            print(f"Error updating timer UI: {e}")
            break
        await asyncio.sleep(1)

# --- Deployment Logic ---
async def run_deployment_async(
    project_id: str, location: str, bucket: str,
    agent_name: str, agent_config: dict, display_name: str, description: str, # Accept edited name/desc
    deploy_button: ui.button, status_area: ui.column,
) -> None:
    """Performs the agent deployment steps asynchronously."""
    deploy_button.disable()
    status_area.clear()

    timer_label = None
    stop_timer_event = asyncio.Event()

    with status_area:
        ui.label(f"Starting deployment for: {agent_name}").classes("text-lg font-semibold")
        progress_label = ui.label("Initializing Vertex AI SDK...")
        spinner = ui.spinner(size="lg", color="primary")
        timer_label = ui.label("Elapsed Time: 00:00").classes("text-sm text-gray-500 mt-1")

    init_success, init_error_msg = await asyncio.to_thread(init_vertex_ai, project_id, location, bucket)

    if not init_success:
        spinner.set_visibility(False)
        with status_area: progress_label.set_text(f"Error: {init_error_msg}")
        ui.notify(f"Vertex AI Initialization Failed: {init_error_msg}", type="negative", multi_line=True, close_button=True)
        deploy_button.enable()
        return

    with status_area:
        progress_label.set_text("Vertex AI Initialized. Importing agent code...")
        ui.notify("Vertex AI Initialized Successfully.", type="positive")

    root_agent, import_error_msg = await get_agent_root_nicegui(agent_config)
    if root_agent is None:
        spinner.set_visibility(False)
        with status_area: progress_label.set_text(f"Error: {import_error_msg}")
        ui.notify(f"Agent Import Failed: {import_error_msg}", type="negative", multi_line=True, close_button=True)
        deploy_button.enable()
        return

    with status_area: progress_label.set_text("Agent code imported. Preparing deployment...")

    adk_app = AdkApp(agent=root_agent, enable_tracing=True)
    agent_specific_reqs = agent_config.get("requirements", [])
    if not isinstance(agent_specific_reqs, list): agent_specific_reqs = []
    combined_requirements = sorted(list(set(_BASE_REQUIREMENTS) | set(agent_specific_reqs)))
    extra_packages = agent_config.get("extra_packages", [])
    if not isinstance(extra_packages, list): extra_packages = []
    # display_name and description are now passed directly as arguments
    # display_name = agent_config.get("ae_display_name", f"{agent_name.replace('_', ' ').title()} Agent") # No longer needed here
    # description = agent_config.get("description", f"Agent: {agent_name}") # No longer needed here

    with status_area: progress_label.set_text("Configuration ready. Deploying ADK to Agent Engine (this may take 2-5 minutes)...")
    print("\n--- Deployment Details ---")
    print(f"Display Name: {display_name}")
    print(f"Description: {description}")
    print("Requirements:"); [print(f"- {req}") for req in combined_requirements]
    print(f"Extra Packages: {extra_packages}")
    print("--------------------------")

    start_time = time.monotonic()
    # Create the timer task but indicate we don't need the task object itself
    _ = asyncio.create_task(update_timer(start_time, timer_label, stop_timer_event, status_area))
    remote_agent = None
    deployment_error = None
    try:
        def sync_create_agent():
            return agent_engines.create(
                adk_app, requirements=combined_requirements, extra_packages=extra_packages,
                display_name=display_name, description=description,
            )
        remote_agent = await asyncio.to_thread(sync_create_agent)
    except Exception as e:
        deployment_error = e
        tb_str = traceback.format_exc()
        print(f"--- Agent creation failed ---\n{tb_str}")
    finally:
        stop_timer_event.set()
        await asyncio.sleep(0.1)
        end_time = time.monotonic()
        duration = end_time - start_time
        duration_str = time.strftime("%M:%S", time.gmtime(duration))
        spinner.set_visibility(False)

        with status_area:
            timer_label.set_text(f"Final Elapsed Time: {duration_str}")

        if remote_agent:
            success_msg = f"Successfully created remote agent: {remote_agent.resource_name}"
            with status_area:
                 progress_label.set_text(f"Deployment Successful! (Duration: {duration_str})")
                 ui.label("Resource Name:").classes("font-semibold mt-2")
                 ui.markdown(f"`{remote_agent.resource_name}`").classes("text-sm")
                 ui.notify(success_msg, type="positive", multi_line=True, close_button=True)
            print(f"--- Agent creation complete ({duration_str}) ---")
        else:
            error_msg = f"Error during agent engine creation: {deployment_error}"
            with status_area:
                 progress_label.set_text(f"Deployment Failed! (Duration: {duration_str})")
                 ui.label("Error Details:").classes("font-semibold mt-2 text-red-600")
                 ui.html(f"<pre class='text-xs p-2 bg-gray-100 dark:bg-gray-800 rounded overflow-auto'>{traceback.format_exc()}</pre>")
                 ui.notify(error_msg, type="negative", multi_line=True, close_button=True)
        deploy_button.enable()

# --- Destruction Logic ---
async def fetch_agents_for_destroy(
    project_id: str, location: str,
    list_container: ui.column, delete_button: ui.button, fetch_button: ui.button,
    page_state: dict # Pass page state for storing fetched agents and selections
) -> None:
    """Fetches agent engines for the destroy tab."""
    if not project_id or not location:
        ui.notify("Please enter both Project ID and Location.", type="warning")
        return

    fetch_button.disable()
    progress_notification = ui.notification(timeout=None, close_button=False)
    list_container.clear()
    # Removed destroy_agent_cards state
    page_state["destroy_agents"] = [] # Reset fetched agents
    page_state["destroy_selected"] = {} # Reset selections
    delete_button.disable()

    init_success, init_error_msg = await asyncio.to_thread(init_vertex_ai, project_id, location) # No bucket needed

    if not init_success:
        progress_notification.dismiss()
        if init_error_msg: ui.notify(init_error_msg, type="negative", multi_line=True, close_button=True)
        fetch_button.enable()
        return

    try:
        ui.notify("Vertex AI initialized successfully.", type="positive")
        progress_notification.message = "Fetching agent engines..."
        progress_notification.spinner = True

        agent_generator = await asyncio.to_thread(agent_engines.list)
        existing_agents = list(agent_generator)

        print(f"Found {len(existing_agents)} agents for destruction list.")
        progress_notification.spinner = False
        progress_notification.message = f"Found {len(existing_agents)} agents."
        list_container.clear()

        if not existing_agents: # Handle case where no agents are found
            page_state["destroy_agents"] = []
            with list_container:
                ui.label("0 Available Agent Engines").classes("text-lg font-semibold mb-2") # Show 0 count
                ui.label(f"No agent engines found in {project_id}/{location}.")
            ui.notify("No agent engines found.", type="info")
        else:
            with list_container:
                page_state["destroy_agents"] = existing_agents # Store fetched agents
                ui.label(f"{len(existing_agents)} Available Agent Engines:").classes("text-lg font-semibold mb-2") # Add count to label
                for agent in existing_agents:
                    resource_name = agent.resource_name
                    create_time_str = agent.create_time.strftime('%Y-%m-%d %H:%M:%S %Z') if agent.create_time else "N/A"
                    update_time_str = agent.update_time.strftime('%Y-%m-%d %H:%M:%S %Z') if agent.update_time else "N/A"
                    description_str = "No description."
                    if hasattr(agent, '_gca_resource') and hasattr(agent._gca_resource, 'description') and agent._gca_resource.description:
                        description_str = agent._gca_resource.description

                    # Create card without border classes
                    card = ui.card().classes("w-full mb-2 p-3")

                    with card: # Add content to the card
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label(f"{agent.display_name}").classes("text-lg font-medium")
                        with ui.column().classes("gap-0 mt-1 text-sm text-gray-600 dark:text-gray-400"):
                            ui.label(f"Resource: {resource_name}")
                            ui.label(f"Description: {description_str}")
                            with ui.row().classes("gap-4 items-center"): # Add items-center
                                ui.label(f"Created: {create_time_str}")
                                ui.label(f"Updated: {update_time_str}")
                        # Add checkbox inside the card context
                        checkbox = ui.checkbox("Select for Deletion")
                        checkbox.bind_value(page_state["destroy_selected"], resource_name) # Bind to selection dict
                        checkbox.classes("absolute top-2 right-2") # Position checkbox
            delete_button.enable()

    except google_exceptions.PermissionDenied:
        msg = f"Permission denied. Ensure the account has 'Vertex AI User' role or necessary permissions in '{project_id}'."
        progress_notification.dismiss(); list_container.clear()
        with list_container: ui.label(msg).classes("text-red-500")
        ui.notify(msg, type="negative", multi_line=True, close_button=True)
    except Exception as e:
        msg = f"Failed to list agent engines: {e}"
        progress_notification.dismiss(); list_container.clear()
        with list_container: ui.label(msg).classes("text-red-500")
        ui.notify(msg, type="negative", multi_line=True, close_button=True)
    finally:
        await asyncio.sleep(2)
        progress_notification.dismiss()
        fetch_button.enable()

async def confirm_and_delete_agents(
    project_id: str, location: str, page_state: dict
) -> None:
    """Shows confirmation dialog and proceeds with deletion if confirmed."""
    selected_map = page_state.get("destroy_selected", {})
    agents_to_delete = [name for name, selected in selected_map.items() if selected]

    if not agents_to_delete:
        ui.notify("No agents selected for deletion.", type="warning")
        return

    with ui.dialog() as dialog, ui.card():
        ui.label("Confirm Deletion").classes("text-xl font-bold")
        ui.label("You are about to permanently delete the following agent(s):")
        for name in agents_to_delete:
            agent_display = name
            for agent in page_state.get("destroy_agents", []):
                if agent.resource_name == name:
                    agent_display = f"{agent.display_name} ({name.split('/')[-1]})" # Show ID too
                    break
            ui.label(f"- {agent_display}")
        ui.label("\nThis action cannot be undone.").classes("font-bold text-red-600")

        with ui.row().classes("mt-4 w-full justify-end"):
            ui.button("Cancel", on_click=dialog.close, color="gray")
            ui.button("Delete Permanently",
                      on_click=lambda: run_actual_deletion(project_id, location, agents_to_delete, page_state, dialog),
                      color="red")
    await dialog

async def run_actual_deletion(
    project_id: str, location: str, resource_names: List[str], page_state: dict, dialog: ui.dialog
) -> None:
    """Performs the actual deletion of agents."""
    dialog.close()

    init_success, init_error_msg = await asyncio.to_thread(init_vertex_ai, project_id, location)
    if not init_success:
        full_msg = f"Failed to re-initialize Vertex AI. Deletion aborted.\nDetails: {init_error_msg}" if init_error_msg else "Failed to re-initialize Vertex AI. Deletion aborted."
        ui.notify(full_msg, type="negative", multi_line=True, close_button=True)
        return

    print("\n--- Deleting Selected Agents ---")
    ui.notify("Starting deletion process...", type="info")
    progress_notification = ui.notification(timeout=None, close_button=False)

    success_count = 0
    fail_count = 0
    failed_agents: List[str] = []

    def delete_single_agent(resource_name_to_delete):
        agent_to_delete = agent_engines.get(resource_name=resource_name_to_delete)
        agent_to_delete.delete(force=True)

    for i, resource_name in enumerate(resource_names):
        try:
            progress_notification.message = f"Deleting {i+1}/{len(resource_names)}: {resource_name.split('/')[-1]}..."
            progress_notification.spinner = True
            print(f"Attempting to delete {resource_name}...")
            await asyncio.to_thread(delete_single_agent, resource_name)
            print(f"Successfully deleted {resource_name}")
            ui.notify(f"Successfully deleted {resource_name.split('/')[-1]}", type="positive")
            success_count += 1
            if resource_name in page_state.get("destroy_selected", {}):
                 del page_state["destroy_selected"][resource_name] # Remove from selection
            # Also remove from the fetched list to update UI implicitly on next fetch
            page_state["destroy_agents"] = [a for a in page_state.get("destroy_agents", []) if a.resource_name != resource_name]

        except Exception as e:
            error_msg = f"Failed to delete {resource_name.split('/')[-1]}: {e}"
            print(error_msg)
            ui.notify(error_msg, type="negative", multi_line=True, close_button=True)
            fail_count += 1
            failed_agents.append(resource_name)
        finally:
            progress_notification.spinner = False

    progress_notification.dismiss()
    print("--- Deletion process finished ---")

    # Show summary dialog
    summary_title = "Deletion Complete" if fail_count == 0 else "Deletion Finished with Errors"
    with ui.dialog() as summary_dialog, ui.card():
        ui.label(summary_title).classes("text-xl font-bold")
        ui.label(f"Successfully deleted: {success_count}")
        ui.label(f"Failed to delete: {fail_count}")
        if failed_agents:
            ui.label("Failed agents:")
            for name in failed_agents: ui.label(f"- {name.split('/')[-1]}")
        with ui.row().classes("mt-4 w-full justify-end"):
            ui.button("OK", on_click=summary_dialog.close)
    await summary_dialog
    # Note: We removed the automatic refresh. User needs to click "Fetch Agents" again.

# --- Registration Logic (Adapted from interactive_register.py) ---

async def fetch_agent_engines_for_register(
    project_id: str, location: str, select_element: ui.select, fetch_button: ui.button, page_state: dict, next_button: ui.button
) -> None:
    """Fetches deployed Agent Engines for the registration tab."""
    if not project_id or not location:
        ui.notify("Please enter Project ID and Location first.", type="warning")
        return

    next_button.disable() # Disable next until fetch is complete and selection is made

    fetch_button.disable()
    select_element.clear()
    select_element.set_value(None)
    page_state["register_agent_engines"] = [] # Clear previous list
    ui.notify("Fetching Agent Engines...", type="info", spinner=True)

    init_success, init_error_msg = await asyncio.to_thread(init_vertex_ai, project_id, location)
    if not init_success:
        ui.notify(f"Vertex AI Init Failed: {init_error_msg}", type="negative", multi_line=True, close_button=True)
        fetch_button.enable()
        return

    try:
        agent_generator = await asyncio.to_thread(agent_engines.list)
        existing_agents = list(agent_generator)
        page_state["register_agent_engines"] = existing_agents # Store fetched agents

        if not existing_agents:
            ui.notify("No deployed Agent Engines found.", type="info")
            select_element.set_options([])
        else:
            options = {}
            for agent in existing_agents:
                create_time_str = agent.create_time.strftime('%Y-%m-%d %H:%M') if agent.create_time else "N/A"
                update_time_str = agent.update_time.strftime('%Y-%m-%d %H:%M') if agent.update_time else "N/A"
                display_text = (f"{agent.display_name} ({agent.resource_name.split('/')[-1]}) | "
                                f"Created: {create_time_str} | Updated: {update_time_str}")
                options[agent.resource_name] = display_text
            select_element.set_options(options)
            ui.notify(f"Found {len(existing_agents)} Agent Engines.", type="positive")

        select_element.set_visibility(True) # Show the select element after fetching
    except Exception as e:
        ui.notify(f"Failed to list Agent Engines: {e}", type="negative", multi_line=True, close_button=True)
        select_element.set_visibility(False) # Keep hidden on error
    finally:
        fetch_button.enable()

async def fetch_agentspace_apps(
    project_id: str, locations: List[str], select_element: ui.select, fetch_button: ui.button, page_state: dict, state_key: str, next_button: Optional[ui.button] = None
) -> None:
    """Fetches Agentspace Apps (Discovery Engine Engines) for selection. next_button is optional."""
    if not project_id or not locations:
        ui.notify("Please provide Project ID and Agentspace Locations.", type="warning")
        return

    if next_button: next_button.disable() # Disable next if provided
    select_element.set_visibility(False) # Hide select while fetching

    if not get_agentspace_apps_from_projectid:
        ui.notify("Error: 'get_agentspace_apps_from_projectid' function not available.", type="negative")
        if next_button: next_button.enable() # Re-enable if error before fetch
        return

    fetch_button.disable()
    select_element.clear()
    select_element.set_value(None)
    page_state[state_key] = [] # Clear previous list (e.g., 'register_agentspaces' or 'deregister_agentspaces')
    locations_display = ", ".join(locations)
    ui.notify(f"Fetching Agentspace Apps in {locations_display}...", type="info", spinner=True)

    try:
        # Run the potentially blocking function in a thread
        project_agentspaces = await asyncio.to_thread(
            get_agentspace_apps_from_projectid, project_id, locations=locations # Pass the list directly
        )
        page_state[state_key] = project_agentspaces # Store fetched apps

        if not project_agentspaces:
            ui.notify("No Agentspace Apps found for the specified locations.", type="info")
            select_element.set_options([])
        else:
            # Use engine_id as the key, store the whole dict as value implicitly? No, need unique key for select
            # Let's use index as key for simplicity, or a composite key
            options = {f"{app['location']}/{app['engine_id']}": f"ID: {app['engine_id']} (Loc: {app['location']}, Tier: {app['tier']})"
                       for app in project_agentspaces}
            select_element.set_options(options)
            ui.notify(f"Found {len(project_agentspaces)} Agentspace Apps.", type="positive")

    except Exception as e:
        select_element.set_visibility(False) # Keep hidden on error
        ui.notify(f"Error fetching Agentspace Apps: {e}", type="negative", multi_line=True, close_button=True)
        print(f"Agentspace fetch error details: {traceback.format_exc()}")
    finally:
        select_element.set_visibility(True) # Show select after fetch attempt (even if empty)
        fetch_button.enable()

def register_agent_with_agentspace_sync(
    project_id: str, project_number: str, agentspace_app: Dict[str, Any],
    agent_engine_resource_name: str, agent_display_name: str, agent_description: str,
    agent_icon_uri: str, default_assistant_name: str = "default_assistant"
) -> Tuple[bool, str]:
    """Synchronous function to register Agent Engine with Agentspace App."""
    print("\n--- Registering Agent Engine with Agentspace (Sync Call) ---")
    agentspace_app_id = agentspace_app['engine_id']
    agentspace_location = agentspace_app['location']

    try:
        # Get ADC credentials within the thread
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        if not credentials.token:
            raise ValueError("Failed to refresh token from ADC.")
        access_token = credentials.token
        print("Successfully obtained access token from ADC.")

        agent_id = re.sub(r'\W+', '_', agent_display_name.lower())[:50]
        print(f"Using Agent Config ID: {agent_id}")

        hostname = f"{agentspace_location}-discoveryengine.googleapis.com" if agentspace_location != "global" else "discoveryengine.googleapis.com"
        assistant_api_endpoint = f"https://{hostname}/v1alpha/projects/{project_number}/locations/{agentspace_location}/collections/default_collection/engines/{agentspace_app_id}/assistants/{default_assistant_name}"

        common_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "x-goog-user-project": project_id,
        }

        new_agent_config_payload = {
            "id": agent_id,
            "displayName": agent_display_name,
            "vertexAiSdkAgentConnectionInfo": {"reasoningEngine": agent_engine_resource_name},
            "toolDescription": agent_description,
            "icon": {"uri": agent_icon_uri if agent_icon_uri and agent_icon_uri != "n/a" else "https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/smart_toy/default/24px.svg"},
        }

        # --- Step 1: Get current assistant configuration ---
        print(f"Fetching current configuration for assistant: {default_assistant_name}...")
        get_response = requests.get(assistant_api_endpoint, headers=common_headers)
        existing_agent_configs = []
        try:
            get_response.raise_for_status()
            current_config = get_response.json()
            existing_agent_configs = current_config.get("agentConfigs", [])
            print(f"Found {len(existing_agent_configs)} existing agent configuration(s).")
        except requests.exceptions.RequestException as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"Assistant '{default_assistant_name}' not found. Will create it with the new agent.")
                existing_agent_configs = [] # Start fresh
            else:
                error_detail = f"Status: {e.response.status_code}, Body: {e.response.text}" if e.response else str(e)
                raise ValueError(f"Error fetching current assistant config: {error_detail}") from e

        # --- Step 2: Combine existing configs with the new one (update if ID exists) ---
        updated_configs = [cfg for cfg in existing_agent_configs if cfg.get("id") != agent_id]
        updated_configs.append(new_agent_config_payload)

        # --- Step 3: Patch the assistant with the combined list ---
        # Only include the field specified in the updateMask in the payload
        patch_payload = {
            "agentConfigs": updated_configs,
        }
        patch_endpoint_with_mask = f"{assistant_api_endpoint}?updateMask=agent_configs"
        print(f"Sending PATCH request to: {patch_endpoint_with_mask}")
        # Print the actual payload being sent (which now only contains agentConfigs)
        print(f"Payload (Combined): {json.dumps(patch_payload, indent=2)}")

        response = requests.patch(patch_endpoint_with_mask, headers=common_headers, data=json.dumps(patch_payload))
        response.raise_for_status()

        print("Successfully registered agent with Agentspace.")
        return True, "Registration successful!"

    except requests.exceptions.RequestException as e:
        error_detail = f"Status: {e.response.status_code}, Body: {e.response.text}" if e.response else str(e)
        msg = f"Agentspace registration API call failed: {error_detail}"
        print(msg)
        return False, msg
    except Exception as e:
        msg = f"An unexpected error occurred during registration: {e}\n{traceback.format_exc()}"
        print(msg)
        return False, msg

# --- Deregistration Logic (Adapted from interactive_deregister.py) ---

async def fetch_registered_agents_for_deregister(
    project_id: str, project_number: str, agentspace_app: Dict[str, Any],
    list_container: ui.column, fetch_button: ui.button, deregister_button: ui.button, page_state: dict,
    assistant_name: str = "default_assistant"
) -> None:
    """Fetches agents currently registered within an Agentspace assistant."""
    if not all([project_id, project_number, agentspace_app]):
        ui.notify("Missing Project ID, Number, or selected Agentspace App.", type="warning")
        return

    fetch_button.disable()
    deregister_button.disable()
    list_container.clear()
    page_state["deregister_registered_agents"] = []
    page_state["deregister_selection"] = {}
    ui.notify(f"Fetching registered agents from assistant '{assistant_name}'...", type="info", spinner=True)

    location = agentspace_app['location']
    app_id = agentspace_app['engine_id']

    hostname = f"{location}-discoveryengine.googleapis.com" if location != "global" else "discoveryengine.googleapis.com"
    assistant_api_endpoint = f"https://{hostname}/v1alpha/projects/{project_number}/locations/{location}/collections/default_collection/engines/{app_id}/assistants/{assistant_name}"

    try:
        # Get credentials and make the API call in a thread
        def get_config_sync():
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            access_token = credentials.token
            if not access_token: raise ValueError("Failed to refresh ADC token.")
            headers = {"Authorization": f"Bearer {access_token}", "x-goog-user-project": project_id}
            response = requests.get(assistant_api_endpoint, headers=headers)
            response.raise_for_status()
            return response.json().get("agentConfigs", [])

        agent_configs = await asyncio.to_thread(get_config_sync)
        page_state["deregister_registered_agents"] = agent_configs

        with list_container:
            if not agent_configs:
                ui.label("No agents found registered in this assistant.")
                ui.notify("No registered agents found.", type="info")
            else:
                ui.label(f"Found {len(agent_configs)} registered agents:").classes("font-semibold")
                for cfg in agent_configs:
                    agent_id = cfg.get("id", "Unknown ID")
                    display_name = cfg.get("displayName", "N/A")
                    engine_link = cfg.get('vertexAiSdkAgentConnectionInfo', {}).get('reasoningEngine', 'N/A')
                    with ui.card().classes("w-full p-2 my-1"):
                        with ui.row().classes("items-center"):
                            # Bind checkbox change to update deregister button state
                            checkbox = ui.checkbox().bind_value(page_state["deregister_selection"], agent_id).classes("mr-2")
                            checkbox.on('update:model-value', lambda: update_deregister_button_state(page_state, deregister_button))
                            with ui.column().classes("gap-0"):
                                ui.label(f"{display_name}").classes("font-medium")
                                ui.label(f"ID: {agent_id}").classes("text-xs text-gray-500")
                                ui.label(f"Engine: {engine_link.split('/')[-1]}").classes("text-xs text-gray-500") # Show only last part
                # Initial check for button state after rendering checkboxes
                update_deregister_button_state(page_state, deregister_button)
                ui.notify(f"Successfully fetched {len(agent_configs)} registered agents.", type="positive")

    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 404:
            msg = f"Assistant '{assistant_name}' not found in Agentspace App '{app_id}'."
            with list_container: ui.label(msg)
            ui.notify(msg, type="warning")
        else:
            error_detail = f"Status: {e.response.status_code}, Body: {e.response.text}" if e.response else str(e)
            msg = f"API Error fetching assistant config: {error_detail}"
            with list_container: ui.label(msg).classes("text-red-500")
            ui.notify(msg, type="negative", multi_line=True, close_button=True)
    except Exception as e:
        msg = f"An unexpected error occurred: {e}"
        with list_container: ui.label(msg).classes("text-red-500")
        ui.notify(msg, type="negative", multi_line=True, close_button=True)
        print(f"Fetch registered agents error: {traceback.format_exc()}")
    finally:
        fetch_button.enable()

def deregister_agents_sync(
    project_id: str, project_number: str, agentspace_app: Dict[str, Any],
    agent_ids_to_remove: List[str], current_configs: List[Dict[str, Any]],
    assistant_name: str = "default_assistant"
) -> Tuple[bool, str]:
    """Synchronous function to deregister agents by patching the assistant."""
    print(f"\n--- Deregistering {len(agent_ids_to_remove)} Agent(s) (Sync Call) ---")
    location = agentspace_app['location']
    app_id = agentspace_app['engine_id']

    hostname = f"{location}-discoveryengine.googleapis.com" if location != "global" else "discoveryengine.googleapis.com"
    patch_endpoint_with_mask = f"https://{hostname}/v1alpha/projects/{project_number}/locations/{location}/collections/default_collection/engines/{app_id}/assistants/{assistant_name}?updateMask=agent_configs"

    updated_configs = [cfg for cfg in current_configs if cfg.get("id") not in agent_ids_to_remove]
    payload = {"agentConfigs": updated_configs}

    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        access_token = credentials.token
        if not access_token: raise ValueError("Failed to refresh ADC token.")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "x-goog-user-project": project_id,
        }

        print(f"Sending PATCH request to: {patch_endpoint_with_mask}")
        print(f"Payload (Agent Configs): {json.dumps(payload['agentConfigs'], indent=2)}")
        response = requests.patch(patch_endpoint_with_mask, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        print("Successfully updated Agentspace assistant configuration.")
        return True, f"Successfully deregistered {len(agent_ids_to_remove)} agent(s)."

    except requests.exceptions.RequestException as e:
        error_detail = f"Status: {e.response.status_code}, Body: {e.response.text}" if e.response else str(e)
        msg = f"API Error during deregistration: {error_detail}"
        print(msg)
        return False, msg
    except Exception as e:
        msg = f"An unexpected error occurred during deregistration: {e}\n{traceback.format_exc()}"
        print(msg)
        return False, msg

# --- Helper function for Deregister Tab ---
def update_deregister_button_state(current_page_state: dict, button: ui.button):
    """Enables/disables the deregister button based on selections."""
    selected_ids = [agent_id for agent_id, selected in current_page_state.get("deregister_selection", {}).items() if selected]
    button.set_enabled(bool(selected_ids))


# --- NiceGUI Page Setup ---
@ui.page("/")
async def main_page(client: Client):
    """Main NiceGUI page combining Deploy and Destroy."""

    # --- Page State (Simplified from client_agent_data) ---
    page_state = {
        # Deploy state
        "selected_agent_key": None,
        "selected_agent_config": None,
        "deploy_radio_group": None, # Moved from main_page scope
        "agent_cards": {},
        "previous_selected_card": None,
        # Destroy state
        "destroy_agents": [], # List of fetched AgentEngine objects
        "destroy_selected": {}, # Dict {resource_name: bool}
        # Register state
        "register_agent_engines": [], # List of fetched AgentEngine objects
        "register_agentspaces": [], # List of fetched Agentspace App dicts
        # Deregister state
        "deregister_agentspaces": [], # List of fetched Agentspace App dicts
        "deregister_registered_agents": [], # List of agentConfig dicts from assistant
        "deregister_selection": {}, # Dict {agent_id: bool}
        "project_number": None, # Store project number for deregister
        "selected_deregister_as_app": None, # Store selected app dict for deregister
    }

    # --- UI Elements ---
    ui.query('body').classes(add='text-base')
    header = ui.header(elevated=True).classes("items-center justify-between")
    with header:
        ui.label("ADK on Agent Engine: Lifecycle Manager v0.1").classes("text-2xl font-bold")

    if IMPORT_ERROR_MESSAGE:
        with ui.card().classes("w-full bg-red-100 dark:bg-red-900"):
            ui.label("Configuration Error").classes("text-xl font-bold text-red-700 dark:text-red-300")
            ui.label(IMPORT_ERROR_MESSAGE).classes("text-red-600 dark:text-red-400")
        return

    # --- Right Drawer for Configuration ---
    with ui.right_drawer(top_corner=True, bottom_corner=True).classes("bg-gray-100 dark:bg-gray-800 p-4 flex flex-col").props("bordered") as right_drawer: # Added flex flex-col
        ui.label("Configuration").classes("text-xl font-semibold mb-4")
        with ui.column().classes("gap-4 w-full grow"): # Added grow
            with ui.card().classes("w-full p-4"):
                ui.label("GCP Settings").classes("text-lg font-semibold mb-2")
                project_input = ui.input("GCP Project ID", value=os.getenv("GOOGLE_CLOUD_PROJECT", "")).props("outlined dense").classes('w-full text-base')
                location_select = ui.select(SUPPORTED_REGIONS, label="Agent Engine GCP Location", value=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")).props("outlined dense").classes('w-full text-base')
                # Agentspace locations - changed to multi-select
                agentspace_locations_options = ["global", "us", "eu"]
                default_agentspace_locations = os.getenv("GOOGLE_CLOUD_LOCATIONS", "global,us").split(',')
                agentspace_locations_select = ui.select(agentspace_locations_options, label="Agentspace Locations", multiple=True, value=default_agentspace_locations).props("outlined dense").classes('w-full text-base')
                # Bucket - primarily for deploy, but keep here for simplicity
                bucket_input = ui.input("GCS Staging Bucket (Deploy)", value=os.getenv("AGENTENGINE_STAGING_BUCKET", "")).props("outlined dense prefix=gs://").classes('w-full text-base')
            
            # Spacer to push the following content to the bottom of this column
            ui.element('div').classes('grow')
            ui.html("Created by Aaron Lind<br>avlind@google.com").classes("text-xs text-gray-500 dark:text-gray-400")

    with header:
        ui.button(on_click=lambda: right_drawer.toggle(), icon='menu').props('flat color=white').classes('ml-auto')

    # --- Main Content with Tabs ---
    with ui.tabs().classes('w-full') as tabs:
        deploy_tab = ui.tab('Deploy', icon='rocket_launch')
        register_tab = ui.tab('Register', icon='assignment') # New Tab
        deregister_tab = ui.tab('Deregister', icon='assignment_return') # New Tab
        destroy_tab = ui.tab('Destroy', icon='delete_forever')

    with ui.tab_panels(tabs, value=deploy_tab).classes('w-full'):
        # --- Deploy Tab Panel (Existing) ---
        with ui.tab_panel(deploy_tab):
            with ui.column().classes("w-full p-4 gap-4"):
                with ui.row().classes("items-center gap-2"): # Use a row to place label and icon together
                    ui.label("Select Agent Configuration to Deploy").classes("text-xl font-semibold")
                    info_icon = ui.icon("info", color="primary").classes("cursor-pointer text-lg")
                    with ui.dialog() as info_dialog, ui.card():
                        ui.label(WEBUI_AGENTDEPLOYMENT_HELPTEXT) # Use the imported constant
                        ui.button("Close", on_click=info_dialog.close).classes("mt-4")
                    info_icon.on("click", info_dialog.open)

                deploy_agent_selection_area = ui.grid(columns=2).classes("w-full gap-2")
                deploy_button = ui.button("Deploy Agent", icon="cloud_upload", on_click=lambda: start_deployment())
                deploy_button.disable()
                deploy_status_area = ui.column().classes("w-full mt-2 p-4 border rounded-lg bg-gray-50 dark:bg-gray-900")
                with deploy_status_area:
                    ui.label("Configure deployment and select an agent.").classes("text-gray-500")

        # --- Register Tab Panel (New) ---
        with ui.tab_panel(register_tab):
            with ui.column().classes("w-full p-4 gap-4"):
                ui.label("Register Deployed Agent Engine with Agentspace").classes("text-xl font-semibold")

                with ui.stepper().props('vertical flat').classes('w-full') as stepper:
                    with ui.step("Select Agent Engine"):
                        ui.label("Choose the deployed Agent Engine you want to register.")
                        # Button first
                        register_fetch_ae_button = ui.button("Fetch Agent Engines", icon="refresh")
                        # Select element below, initially hidden
                        register_ae_select = ui.select(options={}, label="Agent Engine").props("outlined dense").classes("w-full mt-2")
                        register_ae_select.set_visibility(False)
                        with ui.stepper_navigation():
                            register_next_button_step1 = ui.button("Next", on_click=stepper.next)
                            register_next_button_step1.bind_enabled_from(register_ae_select, 'value') # Enable based on selection
                        # Connect button click after elements are defined
                        register_fetch_ae_button.on_click(lambda: fetch_agent_engines_for_register(project_input.value, location_select.value, register_ae_select, register_fetch_ae_button, page_state, register_next_button_step1))

                    with ui.step("Select Agentspace App"):
                        ui.label("Choose the Agentspace App where the agent should appear.")
                        # Button first
                        register_fetch_as_button = ui.button("Fetch Agentspace Apps", icon="refresh")
                        # Select element below, initially hidden
                        register_as_select = ui.select(options={}, label="Agentspace App").props("outlined dense").classes("w-full mt-2")
                        register_as_select.set_visibility(False)
                        with ui.stepper_navigation():
                            ui.button("Back", on_click=stepper.previous, color='gray')
                            register_next_button_step2 = ui.button("Next", on_click=stepper.next)
                            register_next_button_step2.bind_enabled_from(register_as_select, 'value') # Enable based on selection
                        # Connect button click after elements are defined
                        register_fetch_as_button.on_click(lambda: fetch_agentspace_apps(project_input.value, agentspace_locations_select.value, register_as_select, register_fetch_as_button, page_state, 'register_agentspaces', register_next_button_step2))

                    with ui.step("Configure & Register"):
                        ui.label("Confirm the details for registration in the Agentspace UI.")
                        register_display_name_input = ui.input("Agent Display Name").props("outlined dense").classes("w-full")
                        register_description_input = ui.textarea("Agent Description").props("outlined dense").classes("w-full")
                        register_icon_input = ui.input("Icon URI (optional, default: smart_toy)", value="n/a").props("outlined dense").classes("w-full")

                        # Logic to populate defaults when selections change
                        async def update_register_defaults():
                            selected_ae_resource = register_ae_select.value
                            selected_ae = next((ae for ae in page_state.get("register_agent_engines", []) if ae.resource_name == selected_ae_resource), None)
                            if selected_ae:
                                # Try finding matching config in AGENT_CONFIGS
                                config_match = next((cfg for cfg in AGENT_CONFIGS.values() if isinstance(cfg, dict) and cfg.get("ae_display_name") == selected_ae.display_name), None)
                                if config_match:
                                    register_display_name_input.value = config_match.get("as_display_name", selected_ae.display_name)
                                    register_description_input.value = config_match.get("description", f"Agent: {selected_ae.display_name}")
                                    register_icon_input.value = config_match.get("as_uri", "n/a")
                                else: # Fallback to AE details
                                    register_display_name_input.value = selected_ae.display_name
                                    register_description_input.value = f"Agent: {selected_ae.display_name}"
                                    register_icon_input.value = "n/a"
                        ui.timer(0.1, update_register_defaults, once=True) # Trigger once initially
                        register_ae_select.on('update:model-value', update_register_defaults) # Trigger on change

                        register_button = ui.button("Register Agent", icon="app_registration", on_click=lambda: start_registration())
                        register_status_area = ui.column().classes("w-full mt-2 p-2 border rounded bg-gray-50 dark:bg-gray-900 min-h-[50px]")
                        with register_status_area: ui.label("Ready to register.").classes("text-sm text-gray-500")

                        with ui.stepper_navigation():
                            ui.button("Back", on_click=stepper.previous, color='gray')
                            # Register button is outside navigation but logically belongs here

        # --- Deregister Tab Panel (New) ---
        with ui.tab_panel(deregister_tab):
            with ui.column().classes("w-full p-4 gap-4"):
                ui.label("Deregister Agent from Agentspace").classes("text-xl font-semibold")

                # Button first for Agentspace selection
                deregister_fetch_as_button = ui.button("Fetch Agentspace Apps", icon="refresh")
                # Select element below, initially hidden
                deregister_as_select = ui.select(options={}, label="Select Agentspace App").props("outlined dense").classes("w-full mt-2")
                deregister_as_select.set_visibility(False)
                # Connect button click after elements are defined
                deregister_fetch_as_button.on_click(lambda: fetch_agentspace_apps(project_input.value, agentspace_locations_select.value, deregister_as_select, deregister_fetch_as_button, page_state, 'deregister_agentspaces')) # No next button needed here

                with ui.card().classes("w-full mt-2"):
                    ui.label("Registered Agents in Selected App").classes("text-lg font-semibold")
                    with ui.row().classes("items-center gap-2 mb-2"):
                         deregister_fetch_reg_button = ui.button("Fetch Registered Agents", icon="refresh",
                                                                 on_click=lambda: fetch_registered_agents_for_deregister(
                                                                     project_input.value, page_state.get('project_number'), # Need project number
                                                                     page_state.get('selected_deregister_as_app'), # Need selected app dict
                                                                     deregister_list_container, deregister_fetch_reg_button,
                                                                     deregister_button, page_state))
                         deregister_fetch_reg_button.bind_enabled_from(deregister_as_select, 'value', backward=lambda x: bool(x))

                    deregister_list_container = ui.column().classes("w-full")
                    with deregister_list_container: ui.label("Select an Agentspace App and click 'Fetch Registered Agents'.").classes("text-gray-500")

                with ui.row().classes("w-full mt-4 justify-end"):
                    deregister_button = ui.button("Deregister Selected Agents", color="red", icon="delete",
                                                  on_click=lambda: confirm_and_deregister())
                    deregister_button.disable() # Enabled when agents are fetched and selected

                deregister_status_area = ui.column().classes("w-full mt-2 p-2 border rounded bg-gray-50 dark:bg-gray-900 min-h-[50px]")
                with deregister_status_area: ui.label("Ready for deregistration.").classes("text-sm text-gray-500")

        # --- Destroy Tab Panel (Existing) ---
        with ui.tab_panel(destroy_tab):
            with ui.column().classes("w-full p-4 gap-4"):
                fetch_destroy_button = ui.button("Fetch Existing Agent Engines", icon="refresh",
                                                 on_click=lambda: fetch_agents_for_destroy(
                                                     project_input.value, location_select.value,
                                                     destroy_list_container, destroy_delete_button, fetch_destroy_button,
                                                     page_state))
                with ui.card().classes("w-full mt-2"):
                    ui.label("Your Agent Engines").classes("text-lg font-semibold")
                    destroy_list_container = ui.column().classes("w-full")
                    with destroy_list_container:
                        ui.label("Click 'Fetch Existing Agents'.").classes("text-gray-500")
                with ui.row().classes("w-full mt-4 justify-end"):
                    destroy_delete_button = ui.button("Delete Selected Agents", color="red", icon="delete_forever",
                                                      on_click=lambda: confirm_and_delete_agents(
                                                          project_input.value, location_select.value, page_state))
                    destroy_delete_button.disable()

    # --- Logic for Deploy Tab ---
    def handle_deploy_agent_selection(agent_key: str):
        nonlocal page_state
        if not all([project_input.value, location_select.value, bucket_input.value]): # Check bucket too for deploy
             ui.notify("Please configure Project, Location, and Bucket in the side panel first.", type="warning")
             if page_state["deploy_radio_group"]: page_state["deploy_radio_group"].set_value(None)
             return

        if page_state["previous_selected_card"]:
            page_state["previous_selected_card"].classes(remove='border-blue-500 dark:border-blue-400')

        page_state["selected_agent_key"] = agent_key
        page_state["selected_agent_config"] = AGENT_CONFIGS.get(agent_key)

        current_card = page_state["agent_cards"].get(agent_key)
        if current_card:
            current_card.classes(add='border-blue-500 dark:border-blue-400')
            page_state["previous_selected_card"] = current_card

        print(f"Selected agent for deploy: {agent_key}")
        update_deploy_button_state()

    def update_deploy_button_state():
        core_config_ok = project_input.value and location_select.value and bucket_input.value
        agent_config_selected = page_state["selected_agent_key"] is not None
        if core_config_ok and agent_config_selected:
            deploy_button.enable()
        else:
            deploy_button.disable()

    # Populate Deploy Agent Selection Area
    with deploy_agent_selection_area:
        if not AGENT_CONFIGS or "error" in AGENT_CONFIGS:
            ui.label("No agent configurations found or error loading them.").classes("text-red-500")
        else:
            page_state["deploy_radio_group"] = ui.radio(
                [key for key in AGENT_CONFIGS.keys()],
                on_change=lambda e: handle_deploy_agent_selection(e.value)
            ).props("hidden")

            for key, config in AGENT_CONFIGS.items():
                card = ui.card().classes("w-full p-3 cursor-pointer hover:shadow-md border-2 border-transparent")
                page_state["agent_cards"][key] = card
                with card.on('click', lambda k=key: page_state["deploy_radio_group"].set_value(k)):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label(f"{config.get('ae_display_name', key)}").classes("text-lg font-medium")
                    with ui.column().classes("gap-0 mt-1 text-sm text-gray-600 dark:text-gray-400"):
                        ui.label(f"Config Key: {key}")
                        ui.label(f"Engine Name: {config.get('ae_display_name', 'N/A')}")
                        ui.label(f"Description: {config.get('description', 'N/A')}")
                        ui.label(f"Module: {config.get('module_path', 'N/A')}:{config.get('root_variable', 'N/A')}")

    async def start_deployment():
        project = project_input.value
        location = location_select.value
        bucket = bucket_input.value
        agent_key = page_state["selected_agent_key"]
        agent_config = page_state["selected_agent_config"]

        if not all([project, location, bucket, agent_key]):
            ui.notify("Please provide Project ID, Location, Bucket, and select an Agent.", type="warning")
            return
        if not agent_config:
            ui.notify("Internal Error: No agent configuration selected.", type="negative")
            return

        with ui.dialog() as confirm_dialog, ui.card():
            ui.label("Confirm Agent Deployment").classes("text-xl font-bold")
            with ui.column().classes("gap-1 mt-2"): # Use column for better spacing
                ui.label("Project:").classes("font-semibold"); ui.label(f"{project}")
                ui.label("Location:").classes("font-semibold"); ui.label(f"{location}")
                ui.label("Bucket:").classes("font-semibold"); ui.label(f"gs://{bucket}")
                ui.label("Agent Config Key:").classes("font-semibold"); ui.label(f"{agent_key}")

            # --- Editable Name and Description ---
            default_display_name = agent_config.get("ae_display_name", f"{agent_key.replace('_', ' ').title()} Agent")
            default_description = agent_config.get("description", f"Agent: {agent_key}")

            display_name_input = ui.input("Agent Engine Name", value=default_display_name).props("outlined dense").classes("w-full mt-3")
            description_input = ui.textarea("Description", value=default_description).props("outlined dense").classes("w-full mt-2")
            # --- End Editable Fields ---

            ui.label("Proceed with deployment?").classes("mt-4")
            with ui.row().classes("mt-4 w-full justify-end"):
                ui.button("Cancel", on_click=confirm_dialog.close, color="gray")
                ui.button("Deploy", on_click=lambda: (
                    confirm_dialog.close(),
                    asyncio.create_task(run_deployment_async(
                        project, location, bucket, agent_key, agent_config,
                        display_name_input.value, description_input.value, # Pass edited values
                        deploy_button, deploy_status_area
                    ))
                ))
        await confirm_dialog

    # --- Logic for Register Tab ---
    async def start_registration():
        project = project_input.value
        project_num = await get_project_number(project)
        selected_ae_resource = register_ae_select.value
        selected_as_key = register_as_select.value # e.g., "global/12345"
        display_name = register_display_name_input.value
        description = register_description_input.value
        icon_uri = register_icon_input.value

        if not all([project, project_num, selected_ae_resource, selected_as_key, display_name]):
            ui.notify("Missing required fields: Project, Agent Engine, Agentspace App, or Display Name.", type="warning")
            return

        # Find the selected agentspace app dict from the stored list
        selected_as_app = next((app for app in page_state.get("register_agentspaces", [])
                                if f"{app['location']}/{app['engine_id']}" == selected_as_key), None)

        if not selected_as_app:
            ui.notify("Internal Error: Could not find selected Agentspace App details.", type="negative")
            return

        register_button.disable()
        with register_status_area:
            register_status_area.clear()
            ui.label("Registering agent...")
            ui.spinner()

        success, message = await asyncio.to_thread(
            register_agent_with_agentspace_sync,
            project, project_num, selected_as_app, selected_ae_resource,
            display_name, description, icon_uri
        )

        with register_status_area:
            register_status_area.clear()
            if success:
                ui.label(f"Success: {message}")
                ui.notify(message, type="positive")
            else:
                ui.label(f"Error: {message}").classes("text-red-500")
                ui.notify(message, type="negative", multi_line=True, close_button=True)
        register_button.enable()

    # --- Logic for Deregister Tab ---
    async def update_deregister_state():
        # Store project number when project changes
        page_state['project_number'] = await get_project_number(project_input.value)
        # Store selected agentspace app dict when selection changes
        selected_as_key = deregister_as_select.value
        selected_as_app = next((app for app in page_state.get("deregister_agentspaces", [])
                                if f"{app['location']}/{app['engine_id']}" == selected_as_key), None)
        page_state['selected_deregister_as_app'] = selected_as_app
        # Clear previous agent list and selections when app changes
        deregister_list_container.clear()
        with deregister_list_container: ui.label("Select an Agentspace App and click 'Fetch Registered Agents'.").classes("text-gray-500")
        page_state["deregister_registered_agents"] = []
        page_state["deregister_selection"] = {}
        update_deregister_button_state(page_state, deregister_button) # Update button state

    project_input.on('update:model-value', update_deregister_state)
    deregister_as_select.on('update:model-value', update_deregister_state)
    # Checkbox on_change handlers are set dynamically when list is populated

    async def confirm_and_deregister():
        selected_ids = [agent_id for agent_id, selected in page_state.get("deregister_selection", {}).items() if selected]
        if not selected_ids:
            ui.notify("No agents selected for deregistration.", type="warning")
            return

        project = project_input.value
        project_num = page_state.get('project_number')
        selected_as_app = page_state.get('selected_deregister_as_app')
        current_configs = page_state.get("deregister_registered_agents", [])

        if not all([project, project_num, selected_as_app]):
            ui.notify("Missing Project or Agentspace App selection.", type="warning")
            return

        with ui.dialog() as dialog, ui.card():
            ui.label("Confirm Deregistration").classes("text-xl font-bold")
            ui.label("Permanently remove the following agent registrations from the Agentspace App?")
            for agent_id in selected_ids:
                 display_name = next((cfg.get('displayName', agent_id) for cfg in current_configs if cfg.get('id') == agent_id), agent_id)
                 ui.label(f"- {display_name} (ID: {agent_id})")
            ui.label("This does NOT delete the underlying Agent Engine deployment.").classes("mt-2")
            with ui.row().classes("mt-4 w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close, color="gray")
                ui.button("Deregister", color="red", on_click=lambda: run_actual_deregistration(
                    project, project_num, selected_as_app, selected_ids, current_configs, dialog
                ))
        await dialog

    async def run_actual_deregistration(project, project_num, selected_as_app, selected_ids, current_configs, dialog):
        dialog.close()
        deregister_button.disable()
        with deregister_status_area: deregister_status_area.clear(); ui.spinner(); ui.label("Deregistering...")
        success, message = await asyncio.to_thread(deregister_agents_sync, project, project_num, selected_as_app, selected_ids, current_configs)
        with deregister_status_area:
            deregister_status_area.clear()
            if success: ui.label(f"Success: {message}"); ui.notify(message, type="positive")
            else: ui.label(f"Error: {message}").classes("text-red-500"); ui.notify(message, type="negative", multi_line=True)
        # Refresh the list of registered agents after deregistration
        await fetch_registered_agents_for_deregister(project, project_num, selected_as_app, deregister_list_container, deregister_fetch_reg_button, deregister_button, page_state)

# --- Main Execution ---
if __name__ in {"__main__", "__mp_main__"}:
    load_dotenv(override=True)
    # Path setup (ensure deployment_utils is found)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path: sys.path.insert(0, script_dir)
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
    utils_path = os.path.join(script_dir, "deployment_utils")
    if os.path.isdir(utils_path) and utils_path not in sys.path: sys.path.insert(0, utils_path)

    ui.run(title="Agent Manager", favicon="🛠️", dark=True, port=8080) # Changed port
