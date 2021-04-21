from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common import exceptions
import sys
import time
import pandas as pd

def scrape(url,query):
    # Note: replace argument with absolute path to the driver executable.
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome('./sel/chromedriver', chrome_options=chrome_options)
    

    driver.get(url)
    driver.maximize_window()
    time.sleep(5)

    try:
        title = driver.find_element_by_xpath('//*[@id="container"]/h1/yt-formatted-string').text
        comment_section = driver.find_element_by_xpath('//*[@id="comments"]')
    except exceptions.NoSuchElementException:
        error = "Error: Double check selector OR "
        error += "element may not yet be on the screen at the time of the find operation"
        print(error)
    
    driver.execute_script("arguments[0].scrollIntoView();", comment_section)
    time.sleep(7)

    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

        time.sleep(2)

        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

    try:
        username_elems = driver.find_elements_by_xpath('//*[@id="author-text"]')
        comment_elems = driver.find_elements_by_xpath('//*[@id="content-text"]')
        like_elems = driver.find_elements_by_xpath('//*[@id="vote-count-middle"]')
    except exceptions.NoSuchElementException:
        error = "Error: Double check selector OR "
        error += "element may not yet be on the screen at the time of the find operation"
        print(error)    

    user_list = []
    comments_list = []
    like_list = []

    for username, comment, like_elems in zip(username_elems, comment_elems,like_elems):
        user_list.append(username.text)
        comments_list.append(comment.text)
        like_list.append(like_elems.text)


    print("> VIDEO TITLE: " + title + "\n")
    df = pd.DataFrame({'title':title, 'user':user_list, 'comments':comments_list, 'like':like_list})
    df.to_csv('./%s.csv'%query, encoding='utf-8-sig')

    

    driver.close()

if __name__ == "__main__":

    df = pd.read_csv("./channels.txt")

    for i in range(len(df)):
        url = df['query'][i]

        query = df['query'][i].split('=')[1]
        scrape(url,query)
