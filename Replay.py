import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

# Set the path to chromedriver.exe in the current directory
chromedriver_path = os.path.join(os.getcwd(), "chromedriver.exe")

# Define the folder path containing the generated Selenium steps scripts
folder_path = "./generated-scripts"

# Function to execute a Selenium steps script
def execute_script(script_file_path, driver):
    with open(script_file_path, "r") as script_file:
        for line in script_file:
            try:
                # Add an explicit wait for the page to load
                # WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.XPATH, 'your_element_locator_here')))
                exec(line, globals(), locals())
            except Exception as e:
                print(f"Error executing script line in {script_file_path}: {line}")
                print(f"Error message: {e}")

# Function to check for JavaScript errors in the console logs
def check_for_js_errors(driver):
    logs = driver.get_log('browser')
    for log in logs:
        if log['level'] == 'SEVERE' and 'Error' in log['message']:
            return True
    return False

# Initialize the ChromeService
chrome_service = Service(executable_path=chromedriver_path)

# Initialize the Selenium WebDriver
chrome_options = webdriver.ChromeOptions()
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
chrome_options.add_argument("--disable-logging")  # Disable logging to consolezd
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

capabilities = DesiredCapabilities.CHROME
capabilities['goog:loggingPrefs'] = {'browser': 'ALL'}

# Iterate over all files in the folder and run scripts
for filename in os.listdir(folder_path):
    if filename.endswith(".py"):
        script_file_path = os.path.join(folder_path, filename)
        print(f"Executing script: {script_file_path}")

        # Initialize the WebDriver using the service
        driver = webdriver.Chrome(service=chrome_service, options=chrome_options, desired_capabilities=capabilities)

        try:
            # Open the page before executing scripts
            driver.get('https://localhost:7282/')
            execute_script(script_file_path, driver)

            # Check for JavaScript errors in the console logs
            if check_for_js_errors(driver):
                print("JavaScript Error Detected!")
                current_url = driver.current_url
                current_time = time.strftime("%Y%m%d%H%M%S")
                escaped_url = current_url.replace("/", "_").replace(":", "_")

                # Capture screenshot of the page
                screenshot_file = os.path.join(folder_path, f"Error_{escaped_url}_{current_time}.png")
                driver.save_screenshot(screenshot_file)
                print(f"Screenshot saved as {screenshot_file}")

                # Get and log console output
                logs = driver.get_log('browser')
                console_log_file = os.path.join(folder_path, f"Error_{escaped_url}_{current_time}.log")
                with open(console_log_file, "w") as log_file:
                    for log in logs:
                        log_file.write(f"[{log['level']}] - {log['message']}\n")
                print(f"Console output saved as {console_log_file}")

        except Exception as e:
            print(f"Error during script execution: {e}")

        finally:
            driver.quit()  # Close the WebDriver for each script execution

# Stop the ChromeService
chrome_service.stop()
