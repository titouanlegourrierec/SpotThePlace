from typing import List
import os
import re
import time

import pandas as pd
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class StreetViewScraper:
    URL_TEMPLATE = "https://www.google.com/maps/@{lat},{long},10z"  # 10z is the zoom level
    IMAGES_FOLDER = 'images'
    DEFAULT_WAIT_TIME = 3

    # Compile regular expressions
    ANGLE_REGEX = re.compile(r"\d+y")
    ROTATION_REGEX = re.compile(r"(\d+(\.\d+)?h)")

    # CSS Selectors and XPaths
    SELECTORS = {
        "cookie_button": '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[*]/div[1]/div[1]/form[2]/div/div/button/span',
        "streetview_button": "#q2sIQ",
        "streetview_activated": '//*[@id="layer"]/div/div/span[6]',
        "center_screen_button": '//*[@id="QA0Szd"]/div/div/div[1]/div[1]/ul/li[1]/button/div/span'
    }

    def __init__(self, wait_time=DEFAULT_WAIT_TIME, headless=True):
        self.wait_time = wait_time
        self.driver = self._initialize_driver(headless=headless)
        self._create_images_folder()

    def get_streetview_from_coordinates(self, long: float, lat: float, angle: float = 90, rotation: float = 90) -> None:
        """
        Retrieves and saves the streetview image for the specified coordinates.

        Parameters:
            - long (float): The longitude of the location.
            - lat (float): The latitude of the location.
            - angle (float): The angle of view of the camera (default is 90° bigger angle).
            - rotation (float): The rotation of the camera (default is 90°).

        Example:
            >>> scraper = StreetViewScraper()
            >>> scraper.get_streetview_from_coordinates(long=2.2945, lat=48.8584)
        """
        image_url, lat, long = self._get_image_url(long=long, lat=lat)
        if image_url == "":
            return

        if self._is_streetview(image_url):
            image_url = self.ANGLE_REGEX.sub(f"{angle}y", image_url)
            image_url = self.ROTATION_REGEX.sub(f"{rotation}h", image_url)

            self._get_streetview(image_url, os.path.join(self.IMAGES_FOLDER, f"{long}_{lat}.png"))
        else:
            print("The provided coordinates do not have a streetview image.")

        self.driver.quit()

    def get_streetview_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieves and saves streetview images for each row in the provided DataFrame.

        Parameters:
            - df (pd.DataFrame): A DataFrame containing the columns 'country', 'long', and 'lat'.

        Returns:
            - pd.DataFrame: The updated DataFrame with additional columns 'url' and 'image_name'.

        Example:
            >>> scraper = StreetViewScraper()
            >>> df = pd.DataFrame({
            >>>     'country': ['France', 'USA'],
            >>>     'long': [2.2945, -74.0060],
            >>>     'lat': [48.8584, 40.7128]
            >>> })
            >>> df = scraper.get_streetview_from_dataframe(df)
        """
        image_names: List[str] = []
        urls: List[str] = []

        # Get the image URLs
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fetching image URLs"):
            country = row['country']
            long = row['long']
            lat = row['lat']

            current_url, lat, long = self._get_image_url(long=long, lat=lat)
            if self._is_streetview(current_url):
                df.at[index, 'lat'] = lat
                df.at[index, 'long'] = long
                urls.append(current_url)
            else:
                # suppress the row if the coordinates do not have a streetview image
                df.drop(index, inplace=True)

        # Store the image URLs in the DataFrame
        df['url'] = urls

        # Get the streetview images
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fetching streetview images"):
            country = row['country']
            long = row['long']
            lat = row['lat']

            base_url = row['url']
            # Change the y parameter to have a larger angle of view
            base_url = self.ANGLE_REGEX.sub("90y", base_url)

            base_image_name = f"{country}_{long}_{lat}"
            image_names.append(base_image_name)

            for h_param in [90, 180, 270, 360]:
                # Change the h parameter to get different views at 360°
                url = self.ROTATION_REGEX.sub(f"{h_param}h", base_url)
                image_name = base_image_name + f"_{h_param}h"
                self._get_streetview(url, os.path.join(self.IMAGES_FOLDER, image_name + '.png'))

        df['image_name'] = image_names

        self.driver.quit()

        return df

    ##########################################################################################################################

    @staticmethod
    def _initialize_driver(headless: bool = True) -> webdriver.Chrome:
        """
        Initializes and returns a Chrome WebDriver instance with specified options.

        Returns:
            - webdriver.Chrome: An instance of Chrome WebDriver with specified options.
        """
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-search-engine-choice-screen")
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_window_size(1920, 1080)  # Set window size to 1920x1080 for consistent behavior across different screens

        driver.get("https://www.google.com/maps")
        StreetViewScraper._accept_cookies(driver)
        return driver

    @staticmethod
    def _accept_cookies(driver: webdriver.Chrome) -> None:
        """
        Accepts cookies on a webpage using the provided WebDriver instance.

        Parameters:
            - driver (webdriver.Chrome): The WebDriver instance used to interact with the webpage.
        """
        cookie_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, StreetViewScraper.SELECTORS["cookie_button"]))
        )
        cookie_button.click()

    def _create_images_folder(self) -> None:
        """
        Creates a folder to store images if it does not already exist.
        """
        if not os.path.exists(StreetViewScraper.IMAGES_FOLDER):
            os.makedirs(StreetViewScraper.IMAGES_FOLDER)

    def _click_center_of_screen(self) -> None:
        """
        Click at the center of the screen.
        """
        window_size = self.driver.get_window_size()
        width = window_size['width']
        height = window_size['height']
        middle_x = (width // 2) - 50
        middle_y = (height // 2) - 100
        # reference element : maps burger menu
        element = self.driver.find_element(By.XPATH,
                                           self.SELECTORS["center_screen_button"])
        actions = ActionChains(self.driver)
        actions.move_to_element_with_offset(element, middle_x, middle_y).click().perform()

    def _get_image_url(self, long: float, lat: float) -> tuple[str, float, float]:
        """
        Retrieves the URL of the streetview image for the specified coordinates.

        Parameters:
            - long (float): The longitude of the location.
            - lat (float): The latitude of the location.

        Returns:
            - str: The URL of the streetview image.
            - float: The real latitude of the location.
            - float: The real longitude of the location.
        """

        # Go to the specified location
        self.driver.get(self.URL_TEMPLATE.format(long=long, lat=lat))

        try:
            # Verify if we are already in streetview mode
            WebDriverWait(self.driver, 6).until(
                EC.presence_of_element_located((By.XPATH, self.SELECTORS["streetview_activated"]))
            )
        except TimeoutException:
            # Click the streetview button
            try:
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, self.SELECTORS["streetview_button"])
                    )
                )
                element.click()
            except (TimeoutException, NoSuchElementException):
                print(f"Streetview button not found for coordinates: ({lat}, {long})")
                return "", 0.0, 0.0

        # Click at the center of the screen
        self._click_center_of_screen()

        time.sleep(self.wait_time)

        image_url = self.driver.current_url

        coords_part = image_url.split('@')[1].split(',')[:2]
        real_lat = float(coords_part[0])
        real_long = float(coords_part[1])

        return image_url, real_lat, real_long

    def _get_streetview(self,
                        url: str,
                        screenshot_path: str = os.path.join(IMAGES_FOLDER, 'view.png')) -> None:
        """
        Retrieves and saves the streetview image of the specified location.

        Parameters:
            - url (str): The URL of the streetview image.
            - screenshot_path (str): The file path where the screenshot will
            be saved.
        """

        self.driver.get(url)

        time.sleep(self.wait_time)

        self.driver.save_screenshot(screenshot_path)

    def _is_streetview(self, url: str) -> bool:
        """
        Checks if the provided URL is a streetview image.

        Parameters:
            - url (str): The URL to check.

        Returns:
            - bool: True if the URL is a streetview image, False otherwise.
        """
        return "streetview" in url
