# Dictionary mapping agent names (used in --agent_name flag) to their specific configurations.
AGENT_CONFIGS = {
    "tools_agent": {
        "module_path": "agents_gallery.tools_agent.agent",
        "root_variable": "root_agent",  # root_agent is expected entrypoint for ADK
        "requirements": [
            "google-adk (>=0.3.0)",
            "google-cloud-aiplatform[adk, agent_engines]",
            "python-dotenv",
        ],
        "extra_packages": [
            "./agents_gallery/tools_agent",  # Path relative to where interactive_deploy.py is run
        ],
        "ae_display_name": "Tools Demo Agent",
        "as_display_name": "Tools Demo Agent",
        "description": "An agent demonstrating the use of various simple tools.",
    },
    "basic_agent": {
        "module_path": "agents_gallery.basic_agent.agent",
        "root_variable": "root_agent",  # root_agent is expected entrypoint for ADK
        "requirements": [
            "google-adk (>=0.3.0)",
            "google-cloud-aiplatform[adk, agent_engines]",
            "python-dotenv",
        ],
        "extra_packages": [
            "./agents_gallery/basic_agent",  # Path relative to where interactive_deploy.py is run
        ],
        "ae_display_name": "Basic Agent",
        "as_display_name": "Basic Agent",
        "description": "An very basic LLM Agent",
    },
    "loop_agent": {
        "module_path": "agents_gallery.loop_agent.agent",
        "root_variable": "root_agent",  # root_agent is expected entrypoint for ADK
        "requirements": [
            "google-adk (>=0.3.0)",
            "google-cloud-aiplatform[adk, agent_engines]",
            "python-dotenv",
        ],
        "extra_packages": [
            "./agents_gallery/loop_agent",  # Path relative to where interactive_deploy.py is run
        ],
        "ae_display_name": "Debate Team",
        "as_display_name": "Debate Team",
        "description": "A Debate Team agent that demonstrates looping functionality within ADK",
    },
    "search_agent": {
        "module_path": "agents_gallery.search_agent.agent",
        "root_variable": "root_agent",  # root_agent is expected entrypoint for ADK
        "requirements": [
            "google-adk (>=0.3.0)",
            "google-cloud-aiplatform[adk, agent_engines]",
            "python-dotenv",
        ],
        "extra_packages": [
            "./agents_gallery/search_agent",  # Path relative to where interactive_deploy.py is run
        ],
        "ae_display_name": "Basic Search Agent",
        "as_display_name": "Basic Search Agent for AS",
        "description": "A Simple LLM Agent empowered with Google Search Tools",
    },
    "travel_concierge": {
        "module_path": "agents_gallery.travel_concierge.agent",
        "root_variable": "root_agent",  # root_agent is expected entrypoint for ADK
        "requirements": [
            "google-adk (>=0.3.0)",
            "google-cloud-aiplatform[adk, agent_engines]",
            "python-dotenv",
        ],
        "extra_packages": [
            "./agents_gallery/travel_concierge",  # Path relative to where interactive_deploy.py is run
        ],
        "ae_display_name": "Travel Concierge Agent",
        "as_display_name": "Travel Agent for AS",
        "description": "A Travel Concierge to help you plan your trip",
    },
    "reddit_scout_agent": {
        "module_path": "agents_gallery.reddit_scout.agent",
        "root_variable": "root_agent",  # root_agent is expected entrypoint for ADK
        "requirements": [
            "google-adk (>=0.3.0)",
            "google-cloud-aiplatform[adk, agent_engines]",
            "python-dotenv",
            "praw",
        ],
        "extra_packages": [
            "./agents_gallery/reddit_scout",  # Path relative to where interactive_deploy.py is run
        ],
        "ae_display_name": "Reddit Scout",
        "as_display_name": "Reddit Scout for Agentspace",
        "as_uri": "https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/article_person/default/24px.svg",
        "description": "A Reddit scout that searches for the most relevant posts in a given subreddit, or list of subreddits and surfaces them to the user in a conside and consumable manner.",
        
    },
    # "data_science_agent": {
    #     "module_path": "agents_gallery.data_science.agent",
    #     "root_variable": "root_agent",  # root_agent is expected entrypoint for ADK
    #     "requirements": [
    #         "google-adk (>=0.3.0)",
    #         "google-cloud-aiplatform[adk, agent_engines]",
    #         "python-dotenv",
    #         "immutabledict",
    #         "sqlglot",
    #         "db-dtypes",
    #         "regex",
    #         "tabulate",
    #     ],
    #     "extra_packages": [
    #         "./agents_gallery/data_science",  # Path relative to where interactive_deploy.py is run
    #     ],
    #     "ae_display_name": "Data Science Agent",
    #     "as_display_name": "Data Science Agent",
    #     "as_uri": "https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/query_stats/default/24px.svg",
    #     "description": "A Data Science and Data Analytics Multi Agent System",
    # },
}
