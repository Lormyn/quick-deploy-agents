# main_script.py
import json
import os
import re  # For sanitizing agent ID
import sys
import traceback

# Add prompt_toolkit for interactive selection
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import message_dialog, radiolist_dialog

# Import Vertex AI SDK and dotenv
try:
    import vertexai
    from dotenv import load_dotenv
    from vertexai import agent_engines
except ImportError:
    print("Error: Could not import required libraries.")
    print("Please install them using: pip install google-cloud-aiplatform python-dotenv prompt-toolkit")
    sys.exit(1)

# Import requests and resource manager
try:
    import google.auth
    import google.auth.transport.requests
    import requests
    from google.cloud import resourcemanager_v3
except ImportError:
    print("Error: Could not import required libraries for API call.")
    print("Please install them using: pip install requests google-cloud-resource-manager")
    sys.exit(1)

try:
    from deployment_utils.agentspace_lister import get_agentspace_apps_from_projectid
except ImportError:
    print("Error: Could not import 'get_agentspace_apps_from_projectid' from 'agentspace_lister.py'.")
    print("Please ensure 'agentspace_lister.py' exists in the same directory or your Python path.")
    sys.exit(1)

# Import AGENT_CONFIGS from deployment_configs.py
try:
    # Assuming deployment_configs.py is in the same directory or Python path
    from deployment_utils.deployment_configs import AGENT_CONFIGS
except ImportError:
    print("Error: Could not import 'AGENT_CONFIGS' from 'deployment_configs.py'.")
    print(
        "Please ensure 'deployment_configs.py' exists and defines the AGENT_CONFIGS dictionary."
    )
    sys.exit(1)
except AttributeError:
    print(
        "Error: 'deployment_configs.py' does not contain the 'AGENT_CONFIGS' dictionary."
    )
    sys.exit(1)


# Configure logging if you want to see logs from the agentspace_lister module
# logging.basicConfig(level=logging.INFO)

def select_agent_engine(project_id: str, location: str) -> agent_engines.AgentEngine | None:
    """Lists and allows selection of a deployed Agent Engine."""
    print(f"\nFetching deployed Agent Engines in {project_id}/{location}...")
    try:
        existing_agents = agent_engines.list()
        if not existing_agents:
            message_dialog(
                title="No Agent Engines Found",
                text=f"No deployed Agent Engines found in project '{project_id}' and location '{location}'.",
            ).run()
            return None

        # Prepare choices for the radio list dialog
        agent_engine_choices = []
        for agent in existing_agents:
            display_text = (
                f"Name: {agent.display_name}\n"
                f"    ID:   {agent.name}\n" # agent.name is the last part of resource_name
                f"    Resource: {agent.resource_name}"
            )
            agent_engine_choices.append((agent, display_text)) # Return the whole agent object

        selected_agent = radiolist_dialog(
            title="Select Deployed Agent Engine",
            text="Choose an Agent Engine deployment:",
            values=agent_engine_choices,
        ).run()

        return selected_agent # Returns the selected AgentEngine object or None

    except Exception as e:
        tb_str = traceback.format_exc()
        message_dialog(
            title="Error Listing Agent Engines",
            text=f"Failed to list Agent Engines: {e}\n\nTraceback:\n{tb_str}",
        ).run()
        return None

def select_agentspace_app(project_id: str, default_locations: str) -> dict | None:
    """Lists and allows selection of an Agentspace App."""
    # Use environment variable or prompt for locations for Agentspace Apps
    locs_str = prompt(
        "Enter comma-separated locations for Agentspace Apps Lookup (e.g., global): ",
        default=default_locations
    )

    print(f"\nFetching Agentspace Apps for project '{project_id}' in locations: {locs_str}...")

    try:
        project_agentspaces = get_agentspace_apps_from_projectid(project_id, locations=locs_str)

        if not project_agentspaces:
            message_dialog(
                title="No Agentspaces Found",
                text=f"No Agentspace Apps found in project '{project_id}' for the specified locations.",
            ).run()
            return None

        # Prepare choices for the radio list dialog
        agentspace_choices = []
        for i, app_info in enumerate(project_agentspaces):
            # Create a display string for each choice (using previous multi-line format)
            display_text = f"ID:         {app_info['engine_id']}\n" \
                           f"    Location:   {app_info['location']}\n" \
                           f"    Tier:       {app_info['tier']}"
                           # Removed the separator line for this context
            # Store the index and the display text
            agentspace_choices.append((i, display_text)) # Use index as the return value

        selected_index = radiolist_dialog(
            title="Select Agentspace App",
            text="Choose an Agentspace App:",
            values=agentspace_choices,
        ).run()

        if selected_index is None:
            return None # User cancelled

        # Retrieve the full details of the selected app using the index
        selected_app_info = project_agentspaces[selected_index]
        return selected_app_info # Return the dictionary for the selected app

    except Exception as e:
         # Catch any unexpected error during the API call or processing
         message_dialog(
             title="Error Listing Agentspace Apps",
             text=f"An error occurred while fetching Agentspace Apps: {e}",
         ).run()
         print(f"Details: {e}") # Also print to console for debugging
         return None

def get_project_number(project_id: str) -> str | None:
    """Gets the GCP project number from the project ID."""
    try:
        client = resourcemanager_v3.ProjectsClient()
        request = resourcemanager_v3.GetProjectRequest(
            name=f"projects/{project_id}",
        )
        project = client.get_project(request=request)
        # Project number is the numerical part of the name 'projects/123456'
        return project.name.split('/')[-1]
    except Exception as e:
        print(f"Error getting project number for '{project_id}': {e}")
        return None

def register_agent_with_agentspace(
    project_id: str,
    project_number: str,
    agentspace_app_id: str,
    agent_engine_resource_name: str,
    agent_display_name: str,
    agent_description: str,
    agent_icon_uri: str,
    agentspace_location: str, # Location of the target Agentspace App
    credentials, # Use ADC credentials object
    default_assistant_name: str = "default_assistant", # Allow specifying assistant name
) -> bool:
    """Registers the Agent Engine with the specified Agentspace App."""
    print("\n--- Registering Agent Engine with Agentspace ---")

    # Refresh credentials to get a valid access token from ADC
    try:
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        if not credentials.token:
            raise ValueError("Failed to refresh token from ADC.")
        access_token = credentials.token
        print("Successfully obtained access token from ADC.")
    except Exception as e:
        print(f"Error refreshing ADC token: {e}")
        return False

    # Generate a unique ID for the agent config (e.g., from display name)
    # Replace spaces and special chars, ensure it's not too long
    agent_id = re.sub(r'\W+', '_', agent_display_name.lower())[:50]
    print(f"Using Agent Config ID: {agent_id}")

    # Determine the correct API hostname based on location
    if agentspace_location == "global":
        hostname = "discoveryengine.googleapis.com"
    else:
        hostname = f"{agentspace_location}-discoveryengine.googleapis.com" # e.g., us-discoveryengine.googleapis.com

    # Construct the API endpoint for the specified assistant
    assistant_api_endpoint = f"https://{hostname}/v1alpha/projects/{project_number}/locations/{agentspace_location}/collections/default_collection/engines/{agentspace_app_id}/assistants/{default_assistant_name}"

    common_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "x-goog-user-project": project_id,
    }

    # Define the payload to patch the agentConfigs
    payload = {
        "name": f"projects/{project_number}/locations/{agentspace_location}/collections/default_collection/engines/{agentspace_app_id}/assistants/{default_assistant_name}",
        "displayName": default_assistant_name.replace("_", " ").title(), # Derive display name
        "agentConfigs": [{
            "id": agent_id, # The unique ID generated earlier
            "displayName": agent_display_name,
            "vertexAiSdkAgentConnectionInfo": {
                "reasoningEngine": agent_engine_resource_name
            },
            "toolDescription": agent_description,
            "icon": {
                # Use a default icon if the one from config is "n/a" or missing
                "uri": agent_icon_uri if agent_icon_uri != "n/a" else "https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/smart_toy/default/24px.svg"
            },
        }]
    }

    # --- Step 1: Get current assistant configuration ---
    print(f"Fetching current configuration for assistant: {default_assistant_name}...")
    try:
        get_response = requests.get(assistant_api_endpoint, headers=common_headers)
        get_response.raise_for_status()
        current_config = get_response.json()
        existing_agent_configs = current_config.get("agentConfigs", [])
        print(f"Found {len(existing_agent_configs)} existing agent configuration(s).")

    except requests.exceptions.RequestException as e:
        # Handle cases where the assistant might not exist yet (e.g., 404)
        if e.response is not None and e.response.status_code == 404:
            print(f"Assistant '{default_assistant_name}' not found. Will create it with the new agent.")
            existing_agent_configs = []
            # If creating, we don't need the updateMask later
            patch_endpoint = assistant_api_endpoint # Use base endpoint for POST/PUT if needed, but PATCH works too
        else:
            print(f"Error fetching current assistant configuration: {e}")
            if e.response is not None:
                print(f"Response Status Code: {e.response.status_code}")
                try: print(f"Response Body: {get_response.text}")
                except: pass
            return False # Cannot proceed without knowing current state or handling error

    # --- Step 2: Combine existing configs with the new one ---
    new_agent_config = payload["agentConfigs"][0] # The config we defined above
    # Check if an agent with the same ID already exists and update it, otherwise append
    updated_configs = [cfg for cfg in existing_agent_configs if cfg.get("id") != agent_id]
    updated_configs.append(new_agent_config)
    payload["agentConfigs"] = updated_configs # Update payload with the full list

    # --- Step 3: Patch the assistant with the combined list ---
    patch_endpoint_with_mask = f"{assistant_api_endpoint}?updateMask=agent_configs"
    print(f"Sending PATCH request to: {patch_endpoint_with_mask}")
    print(f"Payload (Combined): {json.dumps(payload, indent=2)}")
    try:
        response = requests.patch(patch_endpoint_with_mask, headers=common_headers, data=json.dumps(payload))
        print(response)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        print("Successfully registered agent with Agentspace.")
        # print("Response:", response.json()) # Optional: print full response
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error during Agentspace registration API call: {e}")
        if e.response is not None:
            print(f"Response Status Code: {e.response.status_code}")
            try:
                print(f"Response Body: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Response Body: {e.response.text}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during registration: {e}")
        print(traceback.format_exc())
        return False

def main():
    load_dotenv() # Load environment variables from .env file

    # --- Get Configuration Interactively ---
    default_project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    project_id = prompt("Enter GCP Project ID: ", default=default_project)

    if not project_id:
        message_dialog(
            title="Configuration Error", text="GCP Project ID is required."
        ).run()
        return

    # Location for Agent Engines (Vertex AI SDK init)
    default_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1") # Default to a common region
    location = prompt("Enter GCP Location for Agent Engines: ", default=default_location)

    if not location:
        message_dialog(
            title="Configuration Error", text="GCP Location is required for Agent Engines."
        ).run()
        return

    # --- Initialize Vertex AI SDK ---
    try:
        print("\nInitializing Vertex AI SDK...")
        vertexai.init(project=project_id, location=location)
        print("Vertex AI initialized successfully.")
    except Exception as e:
        message_dialog(
            title="Vertex AI Initialization Error",
            text=f"Error initializing Vertex AI SDK: {e}",
        ).run()
        return

    # --- Select Agent Engine ---
    selected_agent_engine = select_agent_engine(project_id, location)
    if selected_agent_engine is None:
        print("Agent Engine selection cancelled or failed.")
        return

    # --- Select an *existing* Agentspace App (Engine) ---
    # Agentspace Apps are often global, adjust default if needed
    default_agentspace_locs = os.getenv("GOOGLE_CLOUD_LOCATIONS", "global,us") # Changed default
    selected_agentspace_app = select_agentspace_app(project_id, default_agentspace_locs)

    # Registration requires an existing Agentspace App ID, so we must exit if none is selected.
    if selected_agentspace_app is None:
        print("Existing Agentspace App selection cancelled or failed.")
        print("An existing Agentspace App (Engine) ID is required to register the Agent Engine.")
        return

    # --- Determine *intended* Agentspace Details based on selected Agent Engine ---
    selected_ae_display_name = selected_agent_engine.display_name
    default_as_display_name = selected_ae_display_name # Fallback default
    default_description = f"Agentspace for {selected_ae_display_name}" # Fallback default
    default_as_uri = "n/a" # Fallback default for URI

    print(f"\nLooking up configuration for Agent Engine: '{selected_ae_display_name}'...")
    found_config = False
    # Ensure AGENT_CONFIGS is treated as a dictionary
    if not isinstance(AGENT_CONFIGS, dict):
         message_dialog(
             title="Configuration Error",
             text="AGENT_CONFIGS in deployment_configs.py is not a dictionary.",
         ).run()
         # Handle the error appropriately, maybe return or use fallbacks
         print("Error: AGENT_CONFIGS is not a dictionary. Using fallback defaults.")
    else:
        # Proceed with iterating only if it's a dictionary
        for config_name, config_details in AGENT_CONFIGS.items():
            # Check if config_details is a dictionary before accessing keys
            if isinstance(config_details, dict) and config_details.get("ae_display_name") == selected_ae_display_name:
                default_as_display_name = config_details.get("as_display_name", default_as_display_name)
                default_description = config_details.get("description", default_description)
                default_as_uri = config_details.get("as_uri", default_as_uri) # Get the URI or keep fallback
                print(f"Found matching configuration '{config_name}'. Using defaults from deployment_configs.py.")
                found_config = True
                break

    if not found_config:
        print("No matching configuration found in deployment_configs.py. Using fallback defaults.")

    # --- Prompt user to confirm/edit Agentspace details ---
    final_as_display_name = prompt(
        "Enter Agentspace Display Name: ", default=default_as_display_name
    )
    final_description = prompt(
        "Enter Agentspace Description: ", default=default_description
    )

    # --- Get Credentials and Project Number ---
    print("\nFetching required credentials...")
    try:
        # Use ADC for authentication
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except google.auth.exceptions.DefaultCredentialsError as e:
        message_dialog(title="Authentication Error", text=f"Could not get Application Default Credentials: {e}\nRun 'gcloud auth application-default login'.").run()
        return
    except Exception as e:
        message_dialog(title="Authentication Error", text=f"An unexpected error occurred during authentication: {e}").run()
        return

    # Get Project Number (still needed)
    project_number = get_project_number(project_id)
    if not project_number:
        message_dialog(title="Error", text=f"Failed to get project number for {project_id}.").run()
        return

    print("Credentials obtained successfully.")

    # --- Perform Registration ---
    registration_success = register_agent_with_agentspace(
        project_id=project_id,
        project_number=project_number,
        agentspace_app_id=selected_agentspace_app['engine_id'], # Use the selected Agentspace App ID
        agent_engine_resource_name=selected_agent_engine.resource_name,
        agent_display_name=final_as_display_name,
        agent_description=final_description,
        agent_icon_uri=default_as_uri, # Use the URI from config or "n/a"
        agentspace_location=selected_agentspace_app['location'], # Pass the Agentspace App's location
        credentials=credentials # Pass the ADC credentials object
        # default_assistant_name="default_assistant" # Keep using the default unless specified otherwise
    )

    if not registration_success:
        message_dialog(title="Registration Failed", text="Failed to register the agent with Agentspace. Check console logs for details.").run()
        # Decide if you want to return here or continue to print the summary

    # --- Print Final Selection ---
    print("\n--- Final Summary ---")
    print("Selected Agent Engine:")
    print(f"  Display Name: {selected_agent_engine.display_name}")
    print(f"  Resource Name: {selected_agent_engine.resource_name}")
    print(f"  Location:     {selected_agent_engine.location}")

    # Display the selected existing Agentspace App
    print("\nTarget Agentspace App (Engine):")
    print(f"  Project:  {project_id}")
    print(f"  ID:       {selected_agentspace_app['engine_id']}")
    print(f"  Location: {selected_agentspace_app['location']}")
    print(f"  Tier:     {selected_agentspace_app['tier']}")

    print("\nRegistered Agent Configuration Details:")
    print(f"  Display Name: {final_as_display_name}")
    print(f"  Description:  {final_description}")
    print(f"  Icon URI:     {default_as_uri}")
    print(f"  Engine Link:  {selected_agent_engine.resource_name}") # Show the linked engine

    print("\nOperation complete.")


if __name__ == "__main__":
    # Ensure the directory containing this script is in the path
    # This helps if agentspace_lister.py is in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    main()
