from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import pandas as pd
import time

# 配置Chrome選項
options = Options()
options.page_load_strategy = 'eager'
options.add_argument("--disable-notifications")
options.add_experimental_option('detach', True)
options.add_argument("--ignore-certificate-errors")
options.add_argument("--ignore-ssl-errors")
options.add_experimental_option('excludeSwitches', ['enable-automation'])
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(options=options)
driver.get("https://lvr.land.moi.gov.tw/")

driver.maximize_window()

# 切換到第一個frame
driver.switch_to.frame(0)
print(driver.title)

# 點擊交易案件查詢
function1_tag = driver.find_element(By.ID, 'pills-sale-tab')
function1_tag.click()
time.sleep(2)

# 選擇台中市
city_tag = driver.find_element(By.ID, 'p_city')
city_select = Select(city_tag)
city_value = "B"
city_select.select_by_value(city_value)
time.sleep(2)

all_house_data = []

# 選擇不同區
def select_dist(index):
    town_tag = driver.find_element(By.ID, 'p_town')
    town_select = Select(town_tag)
    town_select.select_by_index(index)
    time.sleep(1)
    return town_select.first_selected_option.text

# 抓取每個區的數據
i = 1
while True:
    try:
        dist_name = select_dist(i)
        print(f"正在抓取 {dist_name} 的數據")
    except Exception as g:
        print("沒有下一區了", g)
        break

    # 設置起始年份
    start_year_tag = driver.find_element(By.ID, 'p_startY')
    start_year_select = Select(start_year_tag)
    start_value = "113"
    start_year_select.select_by_value(start_value)
    time.sleep(1)

    # 設置起始月份
    start_month_tag = driver.find_element(By.ID, 'p_startM')
    start_month_select = Select(start_month_tag)
    startM_value = "6"
    start_month_select.select_by_value(startM_value)
    time.sleep(1)

    # 設置結束年份
    end_year_tag = driver.find_element(By.ID, 'p_endY')
    end_year_select = Select(end_year_tag)
    end_value = "113"
    end_year_select.select_by_value(end_value)
    time.sleep(1)

    # 設置結束月份
    end_month_tag = driver.find_element(By.ID, 'p_endM')
    end_month_select = Select(end_month_tag)
    endM_value = "7"
    end_month_select.select_by_value(endM_value)
    time.sleep(1)

    # 點擊搜索按鈕
    buttom_tag = driver.find_element(By.XPATH, "/html/body/section[1]/div[2]/div/form/div[2]/div[13]/div/font[1]/a")
    buttom_tag.click()

    # 等待頁面加載
    time.sleep(10)

    house_data = []

    def get_house_data():
        # 獲取當前頁的數據
        address_elements = driver.find_elements(By.XPATH, '//*[@id="table-item-tbody"]/tr')
        for element in address_elements:
            house_data.append(element.text.split('\n'))

    get_house_data()

    while True:
        try:
            # 嘗試找到並點擊下一頁按鈕
            next_page_button = driver.find_element(By.XPATH, '/html/body/section[2]/div/div/div/div/div[1]/div[2]/div/ul/li[3]/a')
            next_page_button_parent = next_page_button.find_element(By.XPATH, '..')
            if "disabled" in next_page_button_parent.get_attribute("class"):
                print("沒有下一頁了")
                break
            next_page_button.click()
            time.sleep(5)  # 等待頁面加載
            get_house_data()
        except Exception as e:
            print("沒有下一頁了", e)
            break

    all_house_data.extend(house_data)
    i += 1

    # 返回上一頁並重新定位
    driver.back()
    time.sleep(3)
    driver.switch_to.default_content()
    driver.switch_to.frame(0)
    function1_tag = driver.find_element(By.ID, 'pills-sale-tab')
    function1_tag.click()
    time.sleep(2)
    city_tag = driver.find_element(By.ID, 'p_city')
    city_select = Select(city_tag)
    city_select.select_by_value(city_value)
    time.sleep(2)

# 轉換為DataFrame
columns = ["地址", "建案名稱", "總價(萬)", "交易日期", "建坪", "每坪單價", "總坪數", "屋齡", "樓層", "用途", "土地/建物/車位", "房數"]
df = pd.DataFrame(all_house_data, columns=columns[:len(all_house_data[0])])

# 顯示DataFrame
print(df)

# 保存到CSV文件
df.to_csv('real_estate_data.csv', index=False)

driver.close()



