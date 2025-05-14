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

import os

import vertexai
from dotenv import load_dotenv
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import (
    checkboxlist_dialog,
    message_dialog,
    yes_no_dialog,
)
from vertexai import agent_engines

# from utils.adc_utils import get_adc_info_string # No longer needed for confirmation


def run_deletion(project_id: str, location: str) -> None:
    """Lists and deletes selected agent engines."""
    print("\n--- Starting Agent Deletion ---")

    # 1. Initialize Vertex AI SDK
    try:
        print("Initializing Vertex AI SDK...")
        vertexai.init(
            project=project_id,
            location=location,
        )
        print("Vertex AI initialized successfully.")
    except Exception as e:
        message_dialog(
            title="Vertex AI Initialization Error",
            text=f"Error initializing Vertex AI SDK: {e}",
        ).run()
        return

    # 2. List existing agent engines
    try:
        print(f"Fetching agent engines in {project_id}/{location}...")
        # Fetch and immediately convert to list to handle potential generator issues early
        existing_agents_list = list(agent_engines.list())
        print(f"Found {len(existing_agents_list)} agent engine(s).")
    except Exception as e:
        message_dialog(
            title="Error Listing Agents",
            text=f"Failed to list agent engines: {e}",
        ).run() # Keep .run() here
        return

    # Check if empty *after* successful fetch (list conversion happened in try block)
    if not existing_agents_list:
        message_dialog(
            title="No Agents Found",
            text=f"No agent engines found in project '{project_id}' and location '{location}'.",
        ).run()
        print("No agent engines found. Exiting.") # Add console log
        return

    # 3. Prepare choices for selection
    try:
        agent_choices = [
            (agent.resource_name, f"{agent.display_name} ({agent.resource_name})")
            for agent in existing_agents_list # Iterate over the list
        ]
        # Extra check (highly unlikely to fail if existing_agents_list is not empty, but safe)
        if not agent_choices:
             raise ValueError("Failed to prepare choices from fetched agents (agent data might be incomplete).")
    except Exception as e:
        message_dialog(title="Error Preparing Choices", text=f"Could not prepare agent list for display: {e}").run()
        print(f"Error preparing agent choices: {e}")
        return

    # 4. Prompt user to select agents for deletion
    selected_agents = checkboxlist_dialog(
        title="Select Agent Engines to Delete",
        text="Use SPACE to select/deselect Agent Engines. Press ENTER to confirm.",
        values=agent_choices,
    ).run()

    if not selected_agents:
        print("No agents selected for deletion. Operation cancelled.")
        return

    # 5. Confirmation dialog
    confirm_text = "You are about to permanently delete the following agent(s):\n\n"
    for name in selected_agents:
        confirm_text += f"- {name}\n"
    confirm_text += "\nThis action cannot be undone. Proceed?"

    proceed = yes_no_dialog(title="Confirm Deletion", text=confirm_text).run()

    # 6. Perform deletion
    if proceed:
        print("\n--- Deleting Selected Agents ---")
        success_count = 0
        fail_count = 0
        for resource_name in selected_agents:
            try:
                print(f"Deleting {resource_name}...")
                agent_to_delete = agent_engines.get(resource_name)
                agent_to_delete.delete(force=True) # force=True bypasses safety check if agent is used elsewhere
                print(f"Successfully deleted {resource_name}")
                success_count += 1
            except Exception as e:
                print(f"Failed to delete {resource_name}: {e}")
                fail_count += 1
        print("--- Deletion process finished ---")
        summary_text = f"Deletion Summary:\n- Successfully deleted: {success_count}\n- Failed to delete: {fail_count}"
        message_dialog(title="Deletion Complete", text=summary_text).run()
    else:
        print("Deletion cancelled by user.")


def main() -> None:
    load_dotenv()

    # --- Get Configuration Interactively --- # ADC confirmation removed
    project_id = prompt(
        "Enter GCP Project ID: ", default=os.getenv("GOOGLE_CLOUD_PROJECT", "")
    )
    location = prompt(
        "Enter GCP Location: ", default=os.getenv("GOOGLE_CLOUD_LOCATION", "")
    )

    if not project_id or not location:
        message_dialog(
            title="Configuration Error",
            text="Project ID and Location are required.",
        ).run()
        return

    # --- Run the deletion process ---
    run_deletion(project_id, location)


if __name__ == "__main__":
    main()
