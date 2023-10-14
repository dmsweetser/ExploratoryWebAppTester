# This solution requires Python 3.9.7 to run

import os
import time
import random
import numpy as np
import gym
import urllib.parse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import ElementNotInteractableException  # Add this import
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

# Accept the web application URL as user input
web_app_url = input("Enter the web application URL: ")

# Define the available actions, including "select_option" and "enter_date"
actions = ["click", "input_text", "scroll", "select_option", "enter_date"]
num_actions = len(actions)

# Define the maximum number of episodes (testing sessions)
max_episodes = 1

# Define the maximum number of steps per episode
max_steps = 100

# Create a subfolder for generated scripts
subfolder = "./generated-scripts"
if not os.path.exists(subfolder):
    os.makedirs(subfolder)

# Define the path to the /models directory
model_dir = "./models"

# Ensure the /models directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Function to get the domain from a URL
def get_domain(url):
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    return parsed_url.netloc

# Initialize the Selenium WebDriver
chrome_driver_path = os.path.join(os.path.dirname(__file__), "chromedriver.exe")
chrome_service = ChromeService(executable_path=chrome_driver_path)
chrome_service.start()
capabilities = DesiredCapabilities.CHROME
capabilities['goog:loggingPrefs'] = {'browser': 'ALL'}
chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument("--headless")  # Run headless for faster testing
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--remote-debugging-port=9222")
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-notifications")
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--disable-popup-blocking")
chrome_options.add_argument("--disable-logging")  # Disable logging to console
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

driver = webdriver.Chrome(options=chrome_options, desired_capabilities=capabilities)

# Custom Gym environment for the web application
class WebAppEnv(gym.Env):
    def __init__(self, driver):
        super(WebAppEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(max_steps,))  # Adjust the observation space accordingly
        
        self.state = 0  # Initial state
        self.current_step = 0
        self.driver = driver
        self.actions_sequence = []
        self.original_domain = get_domain(web_app_url)

        # Initialize the environment by navigating to the original URL
        self.driver.get(web_app_url)
        self.actions_sequence.append(f"driver.get('{web_app_url}')")

    def reset(self):
        self.state = 0
        self.current_step = 0
        self.driver.get(web_app_url)
        self.actions_sequence = [f"driver.get('{web_app_url}')"]  # Reset actions sequence with the initial navigation
        return self.state

    def step(self, action):
        if self.current_step >= max_steps:
            self.log_actions()  # Log actions even if max steps reached
            return self.state, 0, True, {}  # End of episode

        try:
            # Check if the current domain is different from the original domain
            current_domain = get_domain(self.driver.current_url)
            if current_domain != self.original_domain:
                self.driver.back()  # Navigate back to the previous page
                self.actions_sequence.append(f"driver.back()")
            else:
                # Perform the selected action
                if action == 0:  # Click
                    # Find clickable elements using CSS selectors
                    clickable_elements = self.driver.find_elements(By.CSS_SELECTOR, "a, button")
                    if clickable_elements:
                        # Randomly select a clickable element and click it
                        element_to_click = random.choice(clickable_elements)
                        element_to_click.click()
                        self.actions_sequence.append(f"driver.find_element(By.CSS_SELECTOR, '{element_to_click.get_attribute('css selector')}').click()")

                # Implement the rest of the actions...
                if action == 1:  # Input Text
                    # Find input fields using CSS selectors
                    input_elements = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text'], input[type='password'], input[type='email']")
                    if input_elements:
                        # Randomly select an input field and enter random text
                        element_to_input = random.choice(input_elements)
                        random_text = ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890') for i in range(10))
                        element_to_input.send_keys(random_text)
                        self.actions_sequence.append(f"driver.find_element(By.CSS_SELECTOR, '{element_to_input.get_attribute('css selector')}').send_keys('{random_text}')")

                elif action == 2:  # Scroll
                    # Scroll the page (you can change the scroll amount)
                    scroll_amount = random.randint(1, 3) * 200  # You can adjust the scroll amount as needed
                    self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                    self.actions_sequence.append(f"driver.execute_script('window.scrollBy(0, {scroll_amount});')")

                elif action == 3:  # Select Option
                    # Find select elements using CSS selectors
                    select_elements = self.driver.find_elements(By.CSS_SELECTOR, "select")
                    if select_elements:
                        # Randomly select a select element and choose a random option
                        element_to_select = random.choice(select_elements)
                        select = Select(element_to_select)
                        options = select.options
                        if options:
                            random_option = random.choice(options)
                            select.select_by_value(random_option.get_attribute("value"))
                            self.actions_sequence.append(f"element = driver.find_element(By.CSS_SELECTOR, '{element_to_select.get_attribute('css selector')}'); Select(element).select_by_value('{random_option.get_attribute('value')}')")

                elif action == 4:  # Enter Date
                    # Find date input fields using CSS selectors
                    date_input_elements = self.driver.find_elements(By.CSS_SELECTOR, "input[type='date']")
                    if date_input_elements:
                        # Randomly select a date input field and enter a random date
                        element_to_input = random.choice(date_input_elements)
                        random_date = f"{random.randint(2000, 2023)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                        element_to_input.send_keys(random_date)
                        self.actions_sequence.append(f"driver.find_element(By.CSS_SELECTOR, '{element_to_input.get_attribute('css selector')}').send_keys('{random_date}')")

        except Exception as e:
            print(f"Exception encountered: {e}")
            pass  # Continue to the next action

        # Check for JavaScript errors in the console logs
        if check_for_js_errors(self.driver):
            self.log_errors()
            self.log_actions()  # Log actions when an error is encountered
            reward = self.current_step + 1  # Reward increases with each step to maximize steps
            return self.state, reward, True, {}  # End of episode

        self.current_step += 1
        self.state = self.current_step
        return self.state, 0, False, {}

    def render(self):
        pass

    def close(self):
        self.driver.quit()

    def log_errors(self):
        current_url = self.driver.current_url
        current_time = time.strftime("%Y%m%d%H%M%S")
        escaped_url = current_url.replace("/", "_").replace(":", "_")

        # Capture screenshot of the page
        screenshot_file = os.path.join(subfolder, f"Error_{escaped_url}_{current_time}.png")
        self.driver.save_screenshot(screenshot_file)
        print(f"Screenshot saved as {screenshot_file}")

        # Get and log console output
        logs = self.driver.get_log('browser')
        console_log_file = os.path.join(subfolder, f"Error_{escaped_url}_{current_time}.log")
        with open(console_log_file, "w") as log_file:
            for log in logs:
                log_file.write(f"[{log['level']}] - {log['message']}\n")
        print(f"Console output saved as {console_log_file}")

    def log_actions(self):
        current_url = self.driver.current_url
        current_time = time.strftime("%Y%m%d%H%M%S")
        escaped_url = urllib.parse.quote(current_url, safe='')

        try:
            # Save generated Selenium steps script
            if self.actions_sequence:
                selenium_steps_file = os.path.join(subfolder, f"{escaped_url}_{current_time}.py")

                with open(selenium_steps_file, "w") as actions_file:
                    for action in self.actions_sequence:
                        actions_file.write(f"{action}\n")

                print(f"Generated Selenium steps saved as {selenium_steps_file}")
        except Exception as e:
            print(f"Exception encountered while saving actions: {e}")
            self.log_errors()

# Check if the model file exists in the /models directory
model_path = os.path.join(model_dir, "ppo_web_app_model.zip")
if os.path.exists(model_path):
    # Load the pre-trained reinforcement learning model
    model = PPO.load(model_path)
else :
    env = DummyVecEnv([lambda: WebAppEnv(driver)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_web_app_tensorboard/")

# Function to check for JavaScript errors in the console logs
def check_for_js_errors(driver):
    logs = driver.get_log('browser')
    for log in logs:
        if log['level'] == 'SEVERE' and 'Error' in log['message']:
            return True
    return False

# Train a Proximal Policy Optimization (PPO) agent
model.learn(total_timesteps=max_episodes * max_steps)

# Save the trained model
model.save(model_path)

# Test the trained agent
for episode in range(max_episodes):
    print(f"Episode {episode + 1}/{max_episodes}")
    obs = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            print(f"Total Reward: {total_reward}")
            break

# Close the environment
env.close()
