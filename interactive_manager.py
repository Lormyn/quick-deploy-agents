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
import subprocess
import sys

try:
    from prompt_toolkit.shortcuts import message_dialog, radiolist_dialog
except ImportError:
    print("Error: Could not import 'prompt_toolkit'.")
    print("Please install it using: pip install prompt-toolkit")
    sys.exit(1)

# Define the available actions and their corresponding script filenames
ACTIONS = {
    "deploy": {
        "name": "Deploy New Agent Engine",
        "script": "interactive_deploy.py",
        "description": "Deploy an agent configuration from deployment_configs.py to Vertex AI Agent Engine.",
    },
    "destroy": {
        "name": "Destroy Existing Agent Engine",
        "script": "interactive_destroy.py",
        "description": "Delete one or more deployed Agent Engines from Vertex AI.",
    },
    "register": {
        "name": "Register Agent Engine with Agentspace",
        "script": "interactive_register.py",
        "description": "Register a deployed Agent Engine with an Agentspace App (Assistant).",
    },
    "deregister": {
        "name": "Deregister Agent Engine from Agentspace",
        "script": "interactive_deregister.py",
        "description": "Remove an Agent Engine registration from an Agentspace App (Assistant).",
    },
}

def main():
    """Presents a menu and launches the selected interactive script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    choices = [
        (key, f"{details['name']}\n    ({details['description']})")
        for key, details in ACTIONS.items()
    ]

    selected_action_key = radiolist_dialog(
        title="Agent Lifecycle Manager",
        text="Select an action to perform:",
        values=choices,
    ).run()

    if selected_action_key is None:
        print("Operation cancelled.")
        return

    selected_script_name = ACTIONS[selected_action_key]["script"]
    script_path = os.path.join(script_dir, selected_script_name)

    if not os.path.exists(script_path):
        message_dialog(title="Error", text=f"Script not found: {selected_script_name}").run()
        return

    print(f"\n--- Launching: {selected_script_name} ---")
    # Execute the selected script using the same Python interpreter
    # This allows the launched script to take over the terminal interaction
    subprocess.run([sys.executable, script_path], check=False)
    print(f"\n--- Finished: {selected_script_name} ---")

if __name__ == "__main__":
    main()
