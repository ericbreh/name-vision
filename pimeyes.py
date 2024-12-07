import time
from playwright.sync_api import sync_playwright

url = "https://pimeyes.com/en"


def upload(url, path):
    with sync_playwright() as p:
        results = None
        currenturl = None

        try:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(url)

            allow_all_button = page.wait_for_selector(
                '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll')
            allow_all_button.click()

            upload_button = page.wait_for_selector(
                '//*[@id="hero-section"]/div/div[1]/div/div/div[1]/button[2]'
            )
            upload_button.click()

            file_input = page.query_selector('input[type=file]#file-input')
            file_input.set_input_files(path)

            form_groups = page.query_selector_all('div.permissions div.form-group')
            if not form_groups:
                print("No form groups found")
            for form_group in form_groups:
                print(f"Clicking form group: {form_group}")
                form_group.click()
                
            submit_button = page.wait_for_selector('button:has-text("Start Search")')
            submit_button.click()                

            time.sleep(5)
            currenturl = page.url
            resultsXPATH = '//*[@id="results"]/div/div/div[3]/div/div/div[1]/div/div[1]/button/div/span/span'
            results = page.wait_for_selector(resultsXPATH).inner_text

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            print("Results: ", results)
            print("URL: ", currenturl)
            browser.close()


def main():
    # path = input("Enter path to the image: ")
    path = r"C:\Users\Titan\files-idk\projects\name-vision\IMG_3550.jpg"
    upload(url, path)


if __name__ == "__main__":
    main()
