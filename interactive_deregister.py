#  Copyright (C) 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

# Add prompt_toolkit for interactive selection
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import (
    checkboxlist_dialog,
    message_dialog,
    radiolist_dialog,
    yes_no_dialog,
)

# Import necessary libraries
try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: Could not import dotenv.")
    print("Please install it using: pip install python-dotenv")
    sys.exit(1)

try:
    import google.auth
    import google.auth.transport.requests
    import requests
    from google.cloud import resourcemanager_v3
except ImportError:
    print("Error: Could not import required Google libraries for API calls.")
    print("Please install them using: pip install requests google-auth google-cloud-resource-manager")
    sys.exit(1)

try:
    from deployment_utils.agentspace_lister import get_agentspace_apps_from_projectid
except ImportError:
    print("Error: Could not import 'get_agentspace_apps_from_projectid' from 'deployment_utils.agentspace_lister'.")
    print("Please ensure 'agentspace_lister.py' exists in the 'deployment_utils' directory or your Python path.")
    sys.exit(1)

# --- Helper Functions (Partially reused from interactive_register.py) ---

def select_agentspace_app(project_id: str, default_locations: str) -> Optional[Dict[str, Any]]:
    """Lists and allows selection of an Agentspace App."""
    locs_str = prompt(
        "Enter comma-separated locations for Agentspace Apps Lookup (e.g., global,us): ",
        default=default_locations
    )
    if not locs_str:
        message_dialog(title="Input Error", text="Agentspace locations cannot be empty.").run()
        return None

    print(f"\nFetching Agentspace Apps for project '{project_id}' in locations: {locs_str}...")
    try:
        project_agentspaces = get_agentspace_apps_from_projectid(project_id, locations=locs_str)
        if not project_agentspaces:
            message_dialog(
                title="No Agentspaces Found",
                text=f"No Agentspace Apps found in project '{project_id}' for the specified locations.",
            ).run()
            return None

        agentspace_choices = []
        for i, app_info in enumerate(project_agentspaces):
            display_text = (
                f"ID:         {app_info['engine_id']}\n"
                f"    Location:   {app_info['location']}\n"
                f"    Tier:       {app_info['tier']}"
            )
            agentspace_choices.append((i, display_text))

        selected_index = radiolist_dialog(
            title="Select Agentspace App",
            text="Choose the Agentspace App to manage:",
            values=agentspace_choices,
        ).run()

        if selected_index is None: return None
        return project_agentspaces[selected_index]

    except Exception as e:
         message_dialog(
             title="Error Listing Agentspace Apps",
             text=f"An error occurred while fetching Agentspace Apps: {e}",
         ).run()
         print(f"Details: {traceback.format_exc()}")
         return None

def get_project_number(project_id: str) -> Optional[str]:
    """Gets the GCP project number from the project ID."""
    try:
        client = resourcemanager_v3.ProjectsClient()
        request = resourcemanager_v3.GetProjectRequest(name=f"projects/{project_id}")
        project = client.get_project(request=request)
        return project.name.split('/')[-1]
    except Exception as e:
        print(f"Error getting project number for '{project_id}': {e}")
        return None

def get_agentspace_assistant_config(
    project_number: str, agentspace_app: Dict[str, Any], credentials,
    assistant_name: str = "default_assistant"
) -> Optional[List[Dict[str, Any]]]:
    """Fetches the agentConfigs from the specified Agentspace assistant."""
    location = agentspace_app['location']
    app_id = agentspace_app['engine_id']
    project_id = agentspace_app['project_id'] # Assuming project_id is in the dict

    if location == "global": hostname = "discoveryengine.googleapis.com"
    else: hostname = f"{location}-discoveryengine.googleapis.com"

    assistant_api_endpoint = f"https://{hostname}/v1alpha/projects/{project_number}/locations/{location}/collections/default_collection/engines/{app_id}/assistants/{assistant_name}"

    try:
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        access_token = credentials.token
        if not access_token: raise ValueError("Failed to refresh ADC token.")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "x-goog-user-project": project_id,
        }

        print(f"\nFetching configuration for assistant '{assistant_name}'...")
        response = requests.get(assistant_api_endpoint, headers=headers)
        response.raise_for_status()
        config = response.json()
        agent_configs = config.get("agentConfigs", [])
        print(f"Found {len(agent_configs)} registered agent(s).")
        return agent_configs

    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 404:
            message_dialog(title="Not Found", text=f"Assistant '{assistant_name}' not found in Agentspace App '{app_id}'.").run()
        else:
            message_dialog(title="API Error", text=f"Error fetching assistant configuration: {e}").run()
            try: print(f"Response: {e.response.text}")
            except: pass
        return None
    except Exception as e:
        message_dialog(title="Error", text=f"An unexpected error occurred: {e}").run()
        print(f"Details: {traceback.format_exc()}")
        return None

def deregister_agents_from_agentspace(
    project_number: str, agentspace_app: Dict[str, Any], credentials,
    agent_ids_to_remove: List[str], current_configs: List[Dict[str, Any]],
    assistant_name: str = "default_assistant"
) -> bool:
    """Updates the Agentspace assistant by removing selected agentConfigs."""
    location = agentspace_app['location']
    app_id = agentspace_app['engine_id']
    project_id = agentspace_app['project_id']

    if location == "global": hostname = "discoveryengine.googleapis.com"
    else: hostname = f"{location}-discoveryengine.googleapis.com"

    patch_endpoint_with_mask = f"https://{hostname}/v1alpha/projects/{project_number}/locations/{location}/collections/default_collection/engines/{app_id}/assistants/{assistant_name}?updateMask=agent_configs"

    # Filter out the agents to be removed
    updated_configs = [cfg for cfg in current_configs if cfg.get("id") not in agent_ids_to_remove]

    payload = {
        "agentConfigs": updated_configs
        # Include name and displayName if needed, though updateMask should limit the patch
        # "name": f"projects/{project_number}/locations/{location}/collections/default_collection/engines/{app_id}/assistants/{assistant_name}",
        # "displayName": assistant_name.replace("_", " ").title(),
    }

    try:
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        access_token = credentials.token
        if not access_token: raise ValueError("Failed to refresh ADC token.")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "x-goog-user-project": project_id,
        }

        print(f"\nSending PATCH request to deregister {len(agent_ids_to_remove)} agent(s)...")
        print(f"Payload (Agent Configs): {json.dumps(payload['agentConfigs'], indent=2)}")
        response = requests.patch(patch_endpoint_with_mask, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        print("Successfully updated Agentspace assistant configuration.")
        return True

    except requests.exceptions.RequestException as e:
        message_dialog(title="API Error", text=f"Error updating Agentspace: {e}").run()
        try: print(f"Response: {e.response.text}")
        except: pass
        return False
    except Exception as e:
        message_dialog(title="Error", text=f"An unexpected error occurred during deregistration: {e}").run()
        print(f"Details: {traceback.format_exc()}")
        return False

# --- Main Execution Logic ---
def main():
    load_dotenv()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    utils_path = os.path.join(script_dir, "deployment_utils")
    if os.path.isdir(utils_path) and utils_path not in sys.path: sys.path.insert(0, utils_path)

    default_project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    project_id = prompt("Enter GCP Project ID: ", default=default_project)
    if not project_id: message_dialog(title="Input Error", text="GCP Project ID is required.").run(); return

    project_number = get_project_number(project_id)
    if not project_number: message_dialog(title="Error", text=f"Failed to get project number for {project_id}.").run(); return

    default_agentspace_locs = os.getenv("GOOGLE_CLOUD_LOCATIONS", "global,us")
    selected_agentspace_app = select_agentspace_app(project_id, default_agentspace_locs)
    if not selected_agentspace_app: print("Agentspace App selection cancelled or failed."); return
    selected_agentspace_app['project_id'] = project_id # Add project_id for later use

    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except google.auth.exceptions.DefaultCredentialsError as e:
        message_dialog(title="Authentication Error", text=f"Could not get Application Default Credentials: {e}\nRun 'gcloud auth application-default login'.").run(); return
    except Exception as e:
        message_dialog(title="Authentication Error", text=f"An unexpected error occurred during authentication: {e}").run(); return

    current_agent_configs = get_agentspace_assistant_config(project_number, selected_agentspace_app, credentials)
    if current_agent_configs is None: return # Error handled in function
    if not current_agent_configs: message_dialog(title="No Agents", text="No agents are currently registered in this Agentspace assistant.").run(); return

    agent_choices = [
        (cfg.get("id", f"unknown_id_{i}"), f"Name: {cfg.get('displayName', 'N/A')}\n    ID:   {cfg.get('id', 'N/A')}\n    Engine: {cfg.get('vertexAiSdkAgentConnectionInfo', {}).get('reasoningEngine', 'N/A')}")
        for i, cfg in enumerate(current_agent_configs)
    ]

    selected_agent_ids = checkboxlist_dialog(
        title="Select Agents to Deregister",
        text="Use SPACE to select/deselect agents. Press ENTER to confirm.",
        values=agent_choices,
    ).run()

    if not selected_agent_ids: print("No agents selected for deregistration. Operation cancelled."); return

    confirm_text = "You are about to deregister the following agent(s) from the Agentspace assistant:\n\n"
    for agent_id in selected_agent_ids:
        display_name = next((cfg.get('displayName', agent_id) for cfg in current_agent_configs if cfg.get('id') == agent_id), agent_id)
        confirm_text += f"- {display_name} (ID: {agent_id})\n"
    confirm_text += "\nThis will remove them from the Agentspace UI. The underlying Agent Engine deployment will NOT be deleted.\n\nProceed?"

    proceed = yes_no_dialog(title="Confirm Deregistration", text=confirm_text).run()

    if proceed:
        success = deregister_agents_from_agentspace(
            project_number, selected_agentspace_app, credentials,
            selected_agent_ids, current_agent_configs
        )
        if success: message_dialog(title="Success", text="Selected agent(s) successfully deregistered.").run()
        else: message_dialog(title="Failed", text="Deregistration failed. Check console logs for details.").run()
    else:
        print("Deregistration cancelled by user.")

if __name__ == "__main__":
    main()
