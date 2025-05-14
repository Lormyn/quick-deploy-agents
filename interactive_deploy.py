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

import importlib
import os
import sys
import time
import traceback
from typing import Any, Optional, Tuple

import vertexai
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import message_dialog, radiolist_dialog
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

# --- Configuration Loading ---
try:
    from deployment_utils.constants import SUPPORTED_REGIONS
    from deployment_utils.deployment_configs import AGENT_CONFIGS
except ImportError as e:
    print(
        "Error: Could not import from 'deployment_utils'. "
        f"Ensure 'deployment_configs.py' and 'constants.py' exist. Details: {e}"
    )
    AGENT_CONFIGS = {"error": {"ae_display_name": "Import Error"}}
    SUPPORTED_REGIONS = ["us-central1"]
    IMPORT_ERROR_MESSAGE = (
        "Failed to import 'AGENT_CONFIGS' or 'SUPPORTED_REGIONS' from 'deployment_utils'. "
        "Please ensure 'deployment_configs.py' and 'constants.py' exist in the 'deployment_utils' directory "
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

def get_agent_root(agent_config: dict) -> Tuple[Optional[Any], Optional[str]]:
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
        # parent_dir = os.path.dirname(script_dir) # Incorrect assumption

        # Load the agent-specific .env file if it exists
        agent_directory_name = None
        if module_path:
            parts = module_path.split('.')
            # Expecting format like "agents_gallery.dirname.something"
            if len(parts) >= 2 and parts[0] == "agents_gallery":
                agent_directory_name = parts[1]
                print(f"Derived agent directory name '{agent_directory_name}' from module_path '{module_path}'")
            else:
                print(f"Warning: module_path '{module_path}' does not follow expected 'agents_gallery.dirname.something' pattern.")

        if agent_directory_name:
            agent_dir = os.path.join(script_dir, "agents_gallery", agent_directory_name)
            dotenv_path = os.path.join(agent_dir, ".env")
            if os.path.exists(dotenv_path):
                print(f"Loading environment variables from: {dotenv_path}")
                load_dotenv(dotenv_path=dotenv_path, override=True)
                # Add agent directory to path for internal imports *after* loading .env
                if agent_dir not in sys.path:
                    sys.path.insert(0, agent_dir)
            else:
                print(f"Warning: .env file not found at {dotenv_path}. Agent-specific environment variables not loaded.")
        else:
             print("Warning: Could not determine agent directory name from module_path. Skipping .env load and sys.path addition for agent directory.")

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
        return None, f"An unexpected error occurred during agent import: {e}\n{traceback.format_exc()}"

# --- Deployment Logic ---
def run_deployment(
    project_id: str, location: str, bucket: str,
    agent_name: str, agent_config: dict, display_name: str, description: str,
) -> None:
    """Performs the agent deployment steps."""
    print(f"\n--- Starting deployment for: {agent_name} ---")

    # 1. Initialize Vertex AI
    init_success, init_error_msg = init_vertex_ai(project_id, location, bucket)
    if not init_success:
        message_dialog(title="Error", text=f"Vertex AI Initialization Failed:\n{init_error_msg}").run()
        return

    # 2. Import Agent Code
    root_agent, import_error_msg = get_agent_root(agent_config)
    if root_agent is None:
        message_dialog(title="Error", text=f"Agent Import Failed:\n{import_error_msg}").run()
        return

    # 3. Prepare Deployment Configuration
    adk_app = AdkApp(agent=root_agent, enable_tracing=True)
    agent_specific_reqs = agent_config.get("requirements", [])
    if not isinstance(agent_specific_reqs, list): agent_specific_reqs = []
    combined_requirements = sorted(list(set(_BASE_REQUIREMENTS) | set(agent_specific_reqs)))
    extra_packages = agent_config.get("extra_packages", [])
    if not isinstance(extra_packages, list): extra_packages = []
    # display_name and description are now passed directly as arguments

    print("\n--- Deployment Details ---")
    print(f"Project ID: {project_id}")
    print(f"Location: {location}")
    print(f"Staging Bucket: gs://{bucket}")
    print(f"Agent Config Key: {agent_name}")
    print(f"Agent Engine Name: {display_name}")
    print(f"Description: {description}")
    print("Requirements:")
    for req in combined_requirements: print(f"- {req}")
    print(f"Extra Packages: {extra_packages}")
    print("--------------------------")

    # 4. Deploy to Agent Engine
    print("\nDeploying ADK to Agent Engine (this may take 2-5 minutes)...")
    start_time = time.monotonic()
    remote_agent = None
    deployment_error = None
    try:
        remote_agent = agent_engines.create(
            adk_app, requirements=combined_requirements, extra_packages=extra_packages,
            display_name=display_name, description=description,
        )
    except Exception as e:
        deployment_error = e
        tb_str = traceback.format_exc()
        print(f"--- Agent creation failed ---\n{tb_str}")
    finally:
        end_time = time.monotonic()
        duration = end_time - start_time
        duration_str = time.strftime("%M:%S", time.gmtime(duration))

        if remote_agent:
            success_msg = (
                f"Successfully created remote agent!\n\n"
                f"Resource Name: {remote_agent.resource_name}\n"
                f"Duration: {duration_str}"
            )
            print(f"\n--- Agent creation complete ({duration_str}) ---")
            message_dialog(title="Deployment Successful", text=success_msg).run()
        else:
            error_msg = f"Error during agent engine creation:\n{deployment_error}"
            print(f"\n--- Deployment Failed ({duration_str}) ---")
            message_dialog(title="Deployment Failed", text=error_msg).run()

# --- Main Execution ---
def main():
    """Guides the user through the deployment process."""
    if IMPORT_ERROR_MESSAGE:
        message_dialog(title="Configuration Error", text=IMPORT_ERROR_MESSAGE).run()
        return

    # --- Load Environment Variables ---
    load_dotenv(override=True) # Load from .env in the script's directory first
    # Path setup (ensure deployment_utils is found)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path: sys.path.insert(0, script_dir)
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
    utils_path = os.path.join(script_dir, "deployment_utils")
    if os.path.isdir(utils_path) and utils_path not in sys.path: sys.path.insert(0, utils_path)

    # --- Get Configuration ---
    project_id = prompt(
        "Enter your GCP Project ID: ",
        default=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
    ).strip()
    if not project_id:
        print("Project ID is required.")
        return

    location_completer = WordCompleter(SUPPORTED_REGIONS, ignore_case=True)
    location = prompt(
        "Enter the GCP Location for Agent Engine: ",
        completer=location_completer,
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    ).strip()
    if not location:
        print("Location is required.")
        return
    if location not in SUPPORTED_REGIONS:
        print(f"Warning: Location '{location}' is not in the explicitly supported list: {SUPPORTED_REGIONS}")

    bucket = prompt(
        "Enter the GCS Staging Bucket name (without 'gs://'): ",
        default=os.getenv("AGENTENGINE_STAGING_BUCKET", ""),
    ).strip()
    if not bucket:
        print("Staging Bucket is required.")
        return

    # --- Select Agent ---
    if not AGENT_CONFIGS or "error" in AGENT_CONFIGS:
        message_dialog(title="Error", text="No agent configurations found or error loading them.").run()
        return

    agent_choices = [
        (key, f"{config.get('ae_display_name', key)}\n    ({config.get('description', 'N/A')})")
        for key, config in AGENT_CONFIGS.items()
    ]

    selected_agent_key = radiolist_dialog(
        title="Select Agent to Deploy",
        text="Choose an agent configuration:",
        values=agent_choices,
    ).run()

    if selected_agent_key is None:
        print("Deployment cancelled.")
        return

    selected_config = AGENT_CONFIGS[selected_agent_key]

    # --- Confirm Deployment Details (with editable name/description) ---
    default_display_name = selected_config.get("ae_display_name", f"{selected_agent_key.replace('_', ' ').title()} Agent")
    default_description = selected_config.get("description", f"Agent: {selected_agent_key}")

    print("\n--- Confirm Deployment Details ---")
    print(f"Project ID: {project_id}")
    print(f"Location: {location}")
    print(f"Staging Bucket: gs://{bucket}")
    print(f"Agent Config Key: {selected_agent_key}")

    # Get editable name and description
    final_display_name = prompt(
        "Agent Engine Display Name: ",
        default=default_display_name,
    ).strip()
    final_description = prompt(
        "Agent Engine Description: ",
        default=default_description,
    ).strip()

    print(f"Final Agent Engine Name: {final_display_name}")
    print(f"Final Description: {final_description}")

    confirm = prompt(
        "Proceed with deployment? (y/N): ",
        default="y", # Changed default to 'y'
    ).strip().lower()

    if confirm == 'y':
        run_deployment(
            project_id, location, bucket,
            selected_agent_key, selected_config,
            final_display_name, final_description # Pass the final values
        )
    else:
        print("Deployment cancelled.")

if __name__ == "__main__":
    main()
