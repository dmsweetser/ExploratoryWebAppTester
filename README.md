# Exploratory Web App Tester
_Built in collaboration with generative AI_

## Overview
This Python script is designed for web automation using Selenium and reinforcement learning with Stable Baselines3. It interacts with a web application, performing various actions such as clicking, entering text, scrolling, selecting options, and entering dates. The reinforcement learning model is trained using Proximal Policy Optimization (PPO). It also uses a locally-hosted LLM to provide reasonable sample data for text and date input elements.

## Prerequisites
- Python 3.9.7
- Required Python packages
  - psutil
  - numpy
  - gym
  - stable-baselines3
  - selenium
  - llama_cpp
  - requests

## Installation
1. Clone the repository.
2. Install the required packages by running the provided installation batch file:
   ```
   .\install.bat
   ```
3. Download the relevant ChromeDriver executable and place it in the same directory as the script. You can download it from [ChromeDriver Downloads](https://googlechromelabs.github.io/chrome-for-testing/). The driver for Chrome 120 is provided with the repo.

## Usage
1. Run the script using the provided run batch file:
   ```
   .\run_Explore.bat
   ```
2. Follow the on-screen prompts to enter the web application URL and observe the script's automated interactions. By default, the testing runs headless, but this can be disabled within the script by commenting out this line:
```
chrome_options.add_argument("--headless")  # Run headless for faster testing
```

## Model Training
The script trains a reinforcement learning model using Proximal Policy Optimization (PPO). The trained model is saved to the `models` directory.

## Testing
After training, the script tests the trained agent by running it through a specified number of episodes, providing insights into the agent's performance.

## Folder Structure
- `generated-scripts`: Contains subfolders and files with generated scripts during the automation process.
- `models`: Stores the trained reinforcement learning model.

## Notes
- The script utilizes the llama_cpp library for natural language interactions and reinforcement learning decision making.
- Ensure the ChromeDriver executable is compatible with your Chrome browser version.

## Troubleshooting
- If encountering issues, check the error logs and screenshots generated in the `generated-scripts` directory for debugging purposes.
- If using a different web browser, modify the script accordingly.

## Replay
- The `Explore.py` script generates Selenium scripts as part of its process, and these can be re-run using the `Replay.py` script
- To do this, just run `.\run_Replay.bat`

Feel free to customize the script based on specific web application requirements or extend functionality as needed.
