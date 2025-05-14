#!/bin/bash
#
# This script grants necessary BigQuery permissions to the Vertex AI Agent Engine
# service account within your Google Cloud project.

# --- IMPORTANT: USER ACTION REQUIRED ---
# Please update the following variable with your actual Google Cloud Project ID.
export GOOGLE_CLOUD_PROJECT="csaw-workshop1"

# Retrieve the Project Number from the Project ID
export GOOGLE_CLOUD_PROJECT_NUMBER=$(gcloud projects describe ${GOOGLE_CLOUD_PROJECT} --format='value(projectNumber)')

# Check if the project number was retrieved successfully
[ -z "${GOOGLE_CLOUD_PROJECT_NUMBER}" ] && echo "Error: Could not retrieve project number for ${GOOGLE_CLOUD_PROJECT}" && exit 1

export RE_SA="service-${GOOGLE_CLOUD_PROJECT_NUMBER}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} \
    --member="serviceAccount:${RE_SA}" \
    --condition=None \
    --role="roles/bigquery.user"
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} \
    --member="serviceAccount:${RE_SA}" \
    --condition=None \
    --role="roles/bigquery.dataViewer"