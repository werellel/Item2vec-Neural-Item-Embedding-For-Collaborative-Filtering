1import string
import random

from utils import save_data

def gen_item(item_nums=10, item_list=[]):
    if len(item_list) == 0:
        for idx, item in enumerate(string.ascii_lowercase):
            item_list.append(item)
    else:
        for i in item_list:
            for idx, item in enumerate(string.ascii_lowercase):
                item_list.append(i+item)
            if len(item_list) >= item_nums:
                break
                      
    if len(item_list) >= item_nums:
        return item_list[:item_nums]
    else:
        return gen_item(item_nums, item_list)[:item_nums]
        
def generate_data(user_nums = 1000, item_nums = 100, window_size = 10, click_size = 20):
    item_list = gen_item(item_nums = item_nums)
    user_items_dic = {}
    for uid in range(user_nums):
        rand_idx = random.randrange(window_size//2, item_nums-(window_size//2))
        tmp_item_list = item_list[rand_idx-(window_size//2) : rand_idx + (window_size//2)]
        tmp_click_size = random.randrange(1, click_size)
        items = []
        for _ in range(tmp_click_size):
            items.append(random.choice(tmp_item_list))
        user_items_dic[uid] = tuple(items)

    return user_items_dic

if __name__ == '__main__':
    SAVE_PATH = 'generated_data.txt'
    save_data(generate_data(), SAVE_PATH)
