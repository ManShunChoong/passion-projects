import os
from time import sleep

import yaml
from selenium import webdriver
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

dir_path = os.path.dirname(os.path.realpath(__file__))

# Setup configuration
config_path = f"{dir_path}/config.yaml"
with open(config_path, "r") as stream:
    config = yaml.safe_load(stream)


def get_config(key, exclude=None):
    if exclude is None:
        exclude = []

    def empty_paths(dictionary, current_path, paths=None, excluded_keys=None):
        if paths is None:
            paths = []

        for key, value in dictionary.items():
            if excluded_keys and key in excluded_keys:
                continue
            if isinstance(value, dict):
                paths = empty_paths(value, f'{current_path}["{key}"]', paths)
            elif not value:
                paths.append(f'{current_path}["{key}"]')
        return paths

    empty_keys = empty_paths(
        config[key], current_path=f'["{key}"]', excluded_keys=exclude
    )
    if empty_keys:
        message = "\n".join(f"\tconfig{key}" for key in empty_keys)
        raise ValueError(f"empty values found in {config_path}:\n{message}")
    return config[key]


google_chrome = get_config(key="google_chrome")
profile_path = google_chrome["profile_path"]
profile_name = google_chrome["profile_name"]
zoom = get_config(key="zoom")
meeting_link = zoom["meeting_link"]


def join_meeting():
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-data-dir={profile_path}/{profile_name}")

    service = Service(executable_path=ChromeDriverManager().install())
    try:
        driver = webdriver.Chrome(options=options, service=service)
        print("Setup successfully.")

        # Load Zoom profile page (need to log in)
        sleep(1)
        driver.get(url="https://us05web.zoom.us/profile")

        # Click Google login button
        sleep(0.5)
        button = driver.find_element(By.CLASS_NAME, "login-btn-google")
        button.click()

        # Click account button
        sleep(0.5)
        value = '//*[@id="view_container"]/div/div/div[2]/div/div[1]/div/form/span/section/div/div/div/div/ul/li[1]/div'
        button = driver.find_element(By.XPATH, value)
        button.click()

        # Click profile pic edit button
        sleep(2)
        button = driver.find_element(By.CLASS_NAME, "zm-button--icon")
        button.click()

        # Click profile pic delete button
        sleep(0.5)
        value = '//*[@id="app"]/div[2]/div[1]/div/div[3]/div/div[3]/div/div[1]/button'
        button = driver.find_element(By.XPATH, value)
        driver.execute_script("arguments[0].click();", button)

        # Click profile pic confirm button
        sleep(0.5)
        value = '//*[@id="app"]/div[2]/div[1]/div/div[4]/div/div[3]/span/button[1]'
        button = driver.find_element(By.XPATH, value)
        button.click()

        # Load daily Zoom meeting
        sleep(0.5)
        driver.get(url=meeting_link)

        sleep(5)
        driver.close()
    except InvalidArgumentException:
        print(
            "Chrome browser is being used. Please close all Chrome windows before running."
        )


if __name__ == "__main__":
    join_meeting()
