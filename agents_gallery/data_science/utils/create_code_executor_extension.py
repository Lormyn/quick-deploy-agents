# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import vertexai  # Import vertexai at the top level
from dotenv import find_dotenv, load_dotenv, set_key
from vertexai.preview import extensions

# Define the expected path to the .env file for the data_science agent
# This script is in .../data_science/utils/, so .env should be in .../data_science/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
ENV_FILE_PATH = os.path.join(_AGENT_ROOT_DIR, ".env")

# Load .env if it exists. This makes GOOGLE_CLOUD_PROJECT and other variables available.
if os.path.exists(ENV_FILE_PATH):
    print(f"Loading environment variables from: {ENV_FILE_PATH}")
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)
else:
    # Attempt to find .env using find_dotenv as a fallback, then load if found
    found_env_path = find_dotenv()
    if found_env_path:
        print(f"Loading environment variables using find_dotenv() from: {found_env_path}")
        load_dotenv(dotenv_path=found_env_path, override=True)
        ENV_FILE_PATH = found_env_path # Update ENV_FILE_PATH if found elsewhere
    else:
        print(f"Warning: .env file not found at {ENV_FILE_PATH} or by find_dotenv().")
        print("Required variables like GOOGLE_CLOUD_PROJECT might be missing.")
        print(f"Will attempt to create/update {ENV_FILE_PATH} if extension creation is successful.")

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


def _write_extension_name_to_env(extension_name: str, dotenv_path: str):
    """Writes the extension resource name to the specified .env file."""
    try:
        # Ensure the .env file exists before trying to set a key,
        # or create it if it doesn't.
        if not os.path.exists(dotenv_path):
            print(f"Creating .env file at: {dotenv_path}")
            with open(dotenv_path, "w") as f:
                pass # Create an empty file
        
        set_key(dotenv_path, "CODE_INTERPRETER_EXTENSION_NAME", extension_name)
        print(f"Successfully updated 'CODE_INTERPRETER_EXTENSION_NAME' in {dotenv_path}")
    except Exception as e:
        print(f"Error: Could not write CODE_INTERPRETER_EXTENSION_NAME to {dotenv_path}: {e}")


def create_extension() -> extensions.Extension | None:
    # [START generativeaionvertexai_create_extension]
    # Note: vertexai and extensions are already imported at the top level

    # TODO(developer): Update and un-comment below line if PROJECT_ID is not in .env
    # PROJECT_ID = "your-project-id"
    if not PROJECT_ID:
        print("Error: GOOGLE_CLOUD_PROJECT environment variable not set.")
        print("Please set it in your .env file or environment and try again.")
        return None

    try:
        vertexai.init(project=PROJECT_ID, location="us-central1")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        print("Please ensure your GOOGLE_CLOUD_PROJECT is correct and you have authenticated.")
        return None

    try:
        extension = extensions.Extension.create(
            display_name="Code Interpreter",
            description="This extension generates and executes code in the specified language",
            manifest={
                "name": "code_interpreter_tool",
                "description": "Google Code Interpreter Extension",
                "api_spec": {
                    "open_api_gcs_uri": "gs://vertex-extension-public/code_interpreter.yaml"
                },
                "auth_config": {
                    "google_service_account_config": {},
                    "auth_type": "GOOGLE_SERVICE_ACCOUNT_AUTH",
                },
            },
        )
        print(f"Successfully created extension. Resource name: {extension.resource_name}")
        # Example response:
        # projects/123456789012/locations/us-central1/extensions/12345678901234567

        # Write the extension resource name to the .env file
        _write_extension_name_to_env(extension.resource_name, ENV_FILE_PATH)

        # [END generativeaionvertexai_create_extension]
        return extension
    except Exception as e:
        print(f"Error creating Vertex AI Extension: {e}")
        return None


if __name__ == "__main__":
    # Check if an extension name already exists in the .env file
    existing_extension_name = os.getenv("CODE_INTERPRETER_EXTENSION_NAME")
    proceed_with_creation = True

    if existing_extension_name:
        print(f"An existing CODE_INTERPRETER_EXTENSION_NAME was found in {ENV_FILE_PATH}:")
        print(f"  {existing_extension_name}")
        while True:
            choice = input("Do you want to create a new Extension and overwrite this value? (y/n): ").strip().lower()
            if choice == 'y':
                proceed_with_creation = True
                break
            elif choice == 'n':
                proceed_with_creation = False
                print("Skipping extension creation as per user choice.")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        print("No existing CODE_INTERPRETER_EXTENSION_NAME found in .env file.")

    if proceed_with_creation:
        print("Attempting to create Vertex AI Code Interpreter Extension...")
        created_ext = create_extension()
        if created_ext:
            print(f"Extension creation process complete. Resource name: {created_ext.resource_name}")
        else:
            print("Extension creation failed or was aborted.")
