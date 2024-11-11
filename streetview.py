import time
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains


class StreetViewScraper:
    URL_TEMPLATE = "https://www.google.com/maps/@{long},{lat},3a,21z,0h"
    CSS_SELECTOR_STREETVIEW_BUTTON = "#q2sIQ"
    IMAGES_FOLDER = 'images'
    SCREENSHOT_PATH = os.path.join(IMAGES_FOLDER, 'view.png')

    def __init__(self, long: float, lat: float):
        self.long = long
        self.lat = lat
        self.driver = self.initialize_driver()
        self.create_images_folder()

    @staticmethod
    def initialize_driver() -> webdriver.Chrome:
        """
        Initializes and returns a Chrome WebDriver instance with specified options.

        Returns:
            - webdriver.Chrome: An instance of Chrome WebDriver with specified options.
        """
        chrome_options = Options()
        #chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-search-engine-choice-screen")
        driver = webdriver.Chrome(options=chrome_options)
        driver.maximize_window()

        driver.get("https://www.google.com/maps")
        StreetViewScraper.accept_cookies(driver)
        return driver
    
    @staticmethod
    def accept_cookies(driver : webdriver.Chrome) -> None:
        """
        Accepts cookies on a webpage using the provided WebDriver instance.

        Parameters:
            - driver (webdriver.Chrome): The WebDriver instance used to interact with the webpage.
        """
        cookie_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[*]/div[1]/div[1]/form[2]/div/div/button/span'))
        )
        cookie_button.click()

    def create_images_folder(self) -> None:
        """
        Creates a folder to store images if it does not already exist.
        """
        if not os.path.exists(StreetViewScraper.IMAGES_FOLDER):
            os.makedirs(StreetViewScraper.IMAGES_FOLDER)

    def click_center_of_screen(self) -> None:
        """
        Click at the center of the screen.
        """
        window_size = self.driver.get_window_size()
        width = window_size['width']
        height = window_size['height']
        middle_x = width // 2
        middle_y = height // 2
        actions = ActionChains(self.driver)
        actions.move_by_offset(middle_x, middle_y).click().perform()

    def get_streetview_image(self) -> None:
        """
        Get the streetview image of the specified location.
        """
        
        # Go to the specified location
        self.driver.get(self.URL_TEMPLATE.format(long=self.long, lat=self.lat))

        time.sleep(2)

        # Click the streetview button
        element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, self.CSS_SELECTOR_STREETVIEW_BUTTON))
        )
        element.click()

        # Click at the center of the screen
        self.click_center_of_screen()

        time.sleep(2)

        # Save the screenshot of the streetview
        self.driver.save_screenshot(self.SCREENSHOT_PATH)

        # Quit the driver
        self.driver.quit()