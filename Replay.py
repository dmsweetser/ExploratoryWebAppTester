import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

# Set the path to chromedriver.exe in the current directory
chromedriver_path = os.path.join(os.getcwd(), "chromedriver.exe")

# Initialize the Selenium WebDriver
chrome_service = ChromeService(executable_path=chromedriver_path)
chrome_service.start()
capabilities = DesiredCapabilities.CHROME
capabilities['goog:loggingPrefs'] = {'browser': 'ALL'}
chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument("--headless")  # Run headless for testing
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

# Define the folder path containing the generated Selenium steps scripts
folder_path = "./generated-scripts"

# Function to execute a Selenium steps script
def execute_script(script_file_path):
    with open(script_file_path, "r") as script_file:
        for line in script_file:
            try:
                exec(line)
            except Exception as e:
                print(f"Error executing script line in {script_file_path}: {line}")
                print(f"Error message: {e}")

# Iterate over all files in the folder and run scripts
for filename in os.listdir(folder_path):
    if filename.endswith(".py"):
        script_file_path = os.path.join(folder_path, filename)
        print(f"Executing script: {script_file_path}")

        # Initialize the Selenium WebDriver for each script execution
        driver = webdriver.Chrome(executable_path=chromedriver_path, options=chrome_options, desired_capabilities=capabilities)

        try:
            execute_script(script_file_path)

            # Check for JavaScript errors in the console logs
            logs = driver.get_log('browser')
            for log in logs:
                if log['level'] == 'SEVERE' and 'Error' in log['message']:
                    print("JavaScript Error Detected!")
                    current_url = driver.current_url
                    current_time = time.strftime("%Y%m%d%H%M%S")
                    escaped_url = current_url.replace("/", "_").replace(":", "_")

                    # Capture screenshot of the page
                    screenshot_file = os.path.join(folder_path, f"Error_{escaped_url}_{current_time}.png")
                    driver.save_screenshot(screenshot_file)
                    print(f"Screenshot saved as {screenshot_file}")

                    # Get and log console output
                    console_log_file = os.path.join(folder_path, f"Error_{escaped_url}_{current_time}.log")
                    with open(console_log_file, "w") as log_file:
                        for log in logs:
                            log_file.write(f"[{log['level']}] - {log['message']}\n")
                    print(f"Console output saved as {console_log_file}")

        finally:
            driver.quit()  # Close the WebDriver for each script execution

# Stop the ChromeService
chrome_service.stop()
