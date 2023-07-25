import streamlit as st
import subprocess
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import uuid
from PIL import Image, ImageOps

INDEXES_TO_LABELS_DICT = {
    1: 'Shanghaojia Snow Peas 55g', 10: 'Cheetos Japanese Steak Flavor 90g',
    100: 'Vitality Power 105ml', 101: 'Wangzai Milk Reconstituted Milk 250ml',
    102: 'Yili pure milk 250ml', 103: 'Vita Low Sugar Original Soy Milk 250ml',
    104: 'Baiyi Peanut Milk 250ml', 105: 'Huiyi Original Soy Milk 250ml',
    106: 'Yili Yogurt 250ml', 107: 'Yili breakfast milk 250ml',
    108: 'Daliyuan Longan Lotus Seeds 360g', 109: 'Yinlu Rock Sugar Lily Tremella 280g',
    11: 'Cheetos American Turkey Flavor 90g', 110: 'Xiduoduo Assorted Coconuts 567g',
    111: 'Dole Pineapple Chunks 567g', 112: 'Dole Pineapple Chunks 234g',
    113: 'Yinlu barley red bean porridge 280g', 114: 'Yinlu Lotus Seed Polenta 280g',
    115: 'Yinlu Purple Potato and Purple Rice Porridge 280g',
    116: 'Yinlu Coconut Milk Oatmeal 280g', 117: 'Yinlu Brown Sugar Longan 280g',
    118: 'Merlin Luncheon Meat 340g', 119: 'Pearl River Bridge Black Bean Fish 150g',
    12: 'Shanghaojia Corn Sticks Strawberry Flavor 40g', 120: 'Gulong Original Yellow Croaker 120g',
    121: 'Ayam Brand Coconut Milk 140ml', 122: 'Dove Mango Yogurt Chocolate 42g',
    123: 'Dove Mocha Almond Chocolate 43g', 124: 'Dove Passion Fruit White Chocolate 42g',
    125: 'MM peanut milk chocolate beans 40g', 126: 'MM milk chocolate beans 40g',
    127: 'Hershey Milk Chocolate 40g', 128: "Hershey's Creamy White Chocolate Cookies 40g",
    129: 'Crispy Rice Seaweed White Chocolate 24g', 13: 'Ganyuan Crab Roe Flavor Melon Seeds 75g',
    130: 'Crispy Rice Milk Fragrant White Chocolate 24g',
    131: 'Snickers Peanut Sandwich Chocolate 51g',
    132: 'Snickers Oatmeal Peanut Sandwich Chocolate 40g',
    133: 'Snickers Spicy Peanut Sandwich Chocolate 40g',
    134: 'Shine Mai Fruity Spearmint Flavor 37g', 135: 'Shine Mai Fruity Lemon Flavor 37g',
    136: 'Shine Mint Flavor 21g', 137: 'Shine Mai Grape Flavor 21g',
    138: 'Shine Mai Watermelon Flavor 21g', 139: '50g grape flavor',
    14: 'Huiyi Pistachio 140g', 140: 'Green Arrow Sugar Free Peppermint Jasmine Tea Flavor 34g',
    141: 'Green Arrow 5pcs 15g', 142: 'Bibabu Cotton Bubble Gum Cola Flavor 11g',
    143: 'Bibabu Cotton BNB Grape Flavor 11g', 144: 'Starburst Colorful Original Fruit Flavor 25g',
    145: 'Alpine Caramel Milk Flavored Hard Candy 45g',
    146: 'Alpine Milk Fudge Yellow Peach Yogurt Flavor 47g',
    147: 'Alpine Milk Fudge Blueberry Yogurt Flavor 47g',
    148: 'Wanglaoji Throat Lozenges 28g', 149: 'Yili Milk Tablets Blueberry Flavor 32g',
    15: 'Huiyi Salted Peanuts 350g', 150: 'Dr. Xiong Chewing Candy Strawberry Milk Flavor 52g',
    151: 'Skittles Original Fruit Flavor 45g', 152: 'Baoding Tianyu Aged Rice Vinegar 245ml',
    153: 'Hengshun Balsamic Vinegar 340ml', 154: 'Totole Essence of Chicken 200g',
    155: 'Knorr Mushroom Minced Chicken Soup 41g', 156: 'Huiyi Chili Powder 15g',
    157: 'Huiyi Ginger Powder 15g', 158: 'McCormick Salt & Pepper 20g',
    159: 'Starfish Iodized Refined Salt 400g', 16: 'Huiyi Cashew Nuts 160g',
    160: 'Hengshun cooking wine 500ml', 161: 'Donggu Flavored Soy Sauce 150ml',
    162: 'Donggu Yipin Fresh Soy Sauce 150ml', 163: 'Xinhe June Fresh Soy Sauce 160ml',
    164: 'Listerine Zero Mouthwash 80ml', 165: 'Safeguard Pure White Fragrance Shower Gel 100ml',
    166: 'Mei Tao styling gel water 60ml', 167: 'CLEAR Men Shampoo Vitality Sport Mint 50ml',
    168: 'Blue Moon Fengqing Bailan Laundry Detergent 80g',
    169: 'Colgate Brightening Baking Soda 180g', 17: 'Huiyi Wolfberry 100g',
    170: 'Colgate Iced 180g', 171: 'Shuliang White Teeth 80g', 172: 'Yunnan Baiyao Toothpaste 45g',
    173: 'Shuke baby toothbrush for children', 174: 'Breeze Log Pure Gold 100x3',
    175: 'Clean soft face150x3', 176: 'Banbu 100x3', 177: 'Vader Baby 150x3',
    178: 'Xiangyin minions 150x3', 179: 'Breeze log pure black Yao 150x3',
    18: 'Huiyi Dried Sweet Potato 228g', 180: 'Clean cloud velvet touch 130x3',
    181: 'Shu Jie Meng Printing 120x2', 182: 'Xiangyin Hongyue 130x3',
    183: 'Depot Applewood Flavor 90x4', 184: 'Breeze and toughness pure product 130x3',
    185: 'Goldfish bamboo pulp green 135x3', 186: 'Breeze log pure product 150x2',
    187: 'Clean soft face130x3', 188: 'Vida stereo beauty 110x3',
    189: 'Clean Soft CS Single Pack*', 19: 'Huiyi Thai Dried Mango 80g',
    190: 'Xiangyin Minions Single Pack*', 191: 'Fresh Breeze Single Pack*',
    192: 'Xiangyin tea language package*', 193: 'Breeze texture pure product single bag*',
    194: 'Mickey 1928 Notebook', 195: 'Guangbo Solid Glue 15g', 196: 'bill file bag',
    197: 'Morning light snail correction tape', 198: 'Hongtai Liquid Glue 50g',
    199: 'Maped self-adhesive labels', 2: 'Vegetable garden biscuits 80g',
    20: 'Huiyi Dried Yellow Peach 75g', 200: 'east asian marker pen',
    21: 'Huiyi Lemon Slices 65g', 22: 'Xinjiang Hetian Tan Jujube 454g',
    23: 'Huiyi Shiitake Mushroom 100g', 24: 'Huiyi Dried Longan 500g',
    25: 'Huiyi Tea Tree Mushroom 200g', 26: 'Haoxiong Black Fungus Single Slice 150g',
    27: 'Huiyi Boiled Peanuts 454g', 28: 'Huiyi Day Lily 100g',
    29: 'Chacha Herbal Tea Melon Seeds 150g', 3: 'Shanghaojia Shrimp Chips 40g',
    30: 'Chacha Milk Flavor Melon Seeds 150g', 31: 'Chezai Tea Bag Green Tea 50g',
    32: 'Chezai Tea Bag Black Tea 50g', 33: 'Premium sweet potato flavor 80g',
    34: 'Youlemei Red Bean Milk Tea 65g', 35: 'Huanni Potato Porridge 25g',
    36: 'Jiangzhong Hougu Breakfast Rice Thin 40g',
    37: 'Yonghe Soy Milk Sweet Soy Milk Powder 210g', 38: 'Lipton Lemon Flavored Tea 180g',
    39: 'Quaker Multi Berry Oatmeal 40g', 4: 'Shanghaojia Crab Flavor Yizu 40g',
    40: 'Rongyigu Maijia Black Rice Flavor 30g', 41: 'Rongyigu Maijia Red Bean Flavor 30g',
    42: 'Konno Spicy Beef Noodles 112g', 43: 'Jinye Laotan Sauerkraut Beef Noodles 118g',
    44: 'Konno Braised Beef Noodles 114g', 45: 'Hewei Seafood Flavor 84g',
    46: 'Master Kong White Pepper Meat Bone Noodles 76g', 47: 'Master Kong Spicy Beef Noodles 105g',
    48: 'Master Kang Spicy Garlic Pork Rib Noodles 108g',
    49: 'Master Kong Rattan Pepper Beef Noodles 82g', 5: 'Miao crispy magic charcoal flavor 65g',
    50: 'Huafeng Chicken Three Fresh E-Noodles 87g',
    51: 'Master Kong Black Pepper Steak Noodles 104g',
    52: 'Wugudaochang Braised Beef Noodles 100g',
    53: 'Master Kong Laotan Sauerkraut Beef Noodles 114g',
    54: 'Aji Puff Cookies Mango Pineapple Flavor 60g',
    55: 'Qinglian Blueberry Flavor Sandwich Cake 63g',
    56: 'Qinglian Pineapple Flavor Sandwich Cake 63g',
    57: 'Qinglian Strawberry Flavor Sandwich Cake 63g',
    58: 'Garden Wafer Biscuits Strawberry Flavor 50g',
    59: 'Garden Lemon Wafer Biscuits 50g', 6: 'Panpan BBQ Steak Flavor Cubes 105g',
    60: 'Escher Vanilla Milk Flavor 50g', 61: 'Esther Chocolate Flavor 50g',
    62: 'Hyakuriki Shigeru seaweed flavor 60g', 63: 'Pretz Strawberry Milk Flavor 45g',
    64: 'Nestle Crispy Shark 80g', 65: 'Nabanti Chocolate Flavored Wafers 58g',
    66: 'Guili Mediterranean Style Breadsticks 50g', 67: 'Master Kong Miaofu Chocolate Flavor 48g',
    68: 'Love Folks Record Bread 90g', 69: 'Daliyuan Pie strawberry flavor single pack*',
    7: 'Shanghaojia Shrimp Crackers 40g', 70: 'mini Oreo 55g',
    71: 'Nongfu Mountain Spring Mineral Water 550ml', 72: "C'estbon Mineral Water 555ml",
    73: 'Coca-Cola Zero 500ml', 74: 'Coca-Cola 500ml', 75: 'Pepsi 600ml',
    76: 'Fanta Apple Flavor 500ml', 77: 'Fanta Orange 500ml', 78: 'Sprite 500ml',
    79: 'Heineken Beer 500ml', 8: 'Shanghaojia Onion Rings 40g', 80: 'Budweiser 600ml',
    81: 'Pepsi 330ml', 82: 'Coca-Cola 330ml', 83: 'Wanglaoji 310ml',
    84: 'Chapai Yuzu Green Tea 500ml', 85: 'Cha Pai Rose Lychee Black Tea 500ml',
    86: 'Master Kong Iced Black Tea 250ml', 87: 'Jiaduobao 250ml',
    88: 'RIO Fruit Wine Peach Flavor 275ml', 89: 'RIO Fruit Wine Blue Rose Whiskey Flavor 275ml',
    9: '50g', 90: 'Niulanshan Erguotou 100ml', 91: 'Harbin Beer 330ml',
    92: 'Tsingtao Beer 330ml', 93: 'Snow Beer 330ml', 94: 'Harbin Beer 500ml',
    95: 'KELER beer 500ml', 96: 'Budweiser 500ml', 97: 'QQ Star Quancong Milk 125ml',
    98: 'QQ Star Food Milk 125ml', 99: 'Wahaha AD Calcium Milk 220g'
}

STRINGS_TO_NAMES={
    '1': 'Shanghaojia Snow Peas 55g',
    '10': 'Cheetos Japanese Steak Flavor 90g',
    '100': 'Vitality Power 105ml',
    '101': 'Wangzai Milk Reconstituted Milk 250ml',
    '102': 'Yili pure milk 250ml',
    '103': 'Vita Low Sugar Original Soy Milk 250ml',
    '104': 'Baiyi Peanut Milk 250ml',
    '105': 'Huiyi Original Soy Milk 250ml',
    '106': 'Yili Yogurt 250ml',
    '107': 'Yili breakfast milk 250ml',
    '108': 'Daliyuan Longan Lotus Seeds 360g',
    '109': 'Yinlu Rock Sugar Lily Tremella 280g',
    '11': 'Cheetos American Turkey Flavor 90g',
    '110': 'Xiduoduo Assorted Coconuts 567g',
    '111': 'Dole Pineapple Chunks 567g',
    '112': 'Dole Pineapple Chunks 234g',
    '113': 'Yinlu barley red bean porridge 280g',
    '114': 'Yinlu Lotus Seed Polenta 280g',
    '115': 'Yinlu Purple Potato and Purple Rice Porridge 280g',
    '116': 'Yinlu Coconut Milk Oatmeal 280g',
    '117': 'Yinlu Brown Sugar Longan 280g',
    '118': 'Merlin Luncheon Meat 340g',
    '119': 'Pearl River Bridge Black Bean Fish 150g',
    '12': 'Shanghaojia Corn Sticks Strawberry Flavor 40g',
    '120': 'Gulong Original Yellow Croaker 120g',
    '121': 'Ayam Brand Coconut Milk 140ml',
    '122': 'Dove Mango Yogurt Chocolate 42g',
    '123': 'Dove Mocha Almond Chocolate 43g',
    '124': 'Dove Passion Fruit White Chocolate 42g',
    '125': 'MM peanut milk chocolate beans 40g',
    '126': 'MM milk chocolate beans 40g',
    '127': 'Hershey Milk Chocolate 40g',
    '128': "Hershey's Creamy White Chocolate Cookies 40g",
    '129': 'Crispy Rice Seaweed White Chocolate 24g',
    '13': 'Ganyuan Crab Roe Flavor Melon Seeds 75g',
    '130': 'Crispy Rice Milk Fragrant White Chocolate 24g',
    '131': 'Snickers Peanut Sandwich Chocolate 51g',
    '132': 'Snickers Oatmeal Peanut Sandwich Chocolate 40g',
    '133': 'Snickers Spicy Peanut Sandwich Chocolate 40g',
    '134': 'Shine Mai Fruity Spearmint Flavor 37g',
    '135': 'Shine Mai Fruity Lemon Flavor 37g',
    '136': 'Shine Mint Flavor 21g',
    '137': 'Shine Mai Grape Flavor 21g',
    '138': 'Shine Mai Watermelon Flavor 21g',
    '139': '50g grape flavor',
    '14': 'Huiyi Pistachio 140g',
    '140': 'Green Arrow Sugar Free Peppermint Jasmine Tea Flavor 34g',
    '141': 'Green Arrow 5pcs 15g',
    '142': 'Bibabu Cotton Bubble Gum Cola Flavor 11g',
    '143': 'Bibabu Cotton BNB Grape Flavor 11g',
    '144': 'Starburst Colorful Original Fruit Flavor 25g',
    '145': 'Alpine Caramel Milk Flavored Hard Candy 45g',
    '146': 'Alpine Milk Fudge Yellow Peach Yogurt Flavor 47g',
    '147': 'Alpine Milk Fudge Blueberry Yogurt Flavor 47g',
    '148': 'Wanglaoji Throat Lozenges 28g',
    '149': 'Yili Milk Tablets Blueberry Flavor 32g',
    '15': 'Huiyi Salted Peanuts 350g',
    '150': 'Dr. Xiong Chewing Candy Strawberry Milk Flavor 52g',
    '151': 'Skittles Original Fruit Flavor 45g',
    '152': 'Baoding Tianyu Aged Rice Vinegar 245ml',
    '153': 'Hengshun Balsamic Vinegar 340ml',
    '154': 'Totole Essence of Chicken 200g',
    '155': 'Knorr Mushroom Minced Chicken Soup 41g',
    '156': 'Huiyi Chili Powder 15g',
    '157': 'Huiyi Ginger Powder 15g',
    '158': 'McCormick Salt & Pepper 20g',
    '159': 'Starfish Iodized Refined Salt 400g',
    '16': 'Huiyi Cashew Nuts 160g',
    '160': 'Hengshun cooking wine 500ml',
    '161': 'Donggu Flavored Soy Sauce 150ml',
    '162': 'Donggu Yipin Fresh Soy Sauce 150ml',
    '163': 'Xinhe June Fresh Soy Sauce 160ml',
    '164': 'Listerine Zero Mouthwash 80ml',
    '165': 'Safeguard Pure White Fragrance Shower Gel 100ml',
    '166': 'Mei Tao styling gel water 60ml',
    '167': 'CLEAR Men Shampoo Vitality Sport Mint 50ml',
    '168': 'Blue Moon Fengqing Bailan Laundry Detergent 80g',
    '169': 'Colgate Brightening Baking Soda 180g',
    '17': 'Huiyi Wolfberry 100g',
    '170': 'Colgate Iced 180g',
    '171': 'Shuliang White Teeth 80g',
    '172': 'Yunnan Baiyao Toothpaste 45g',
    '173': 'Shuke baby toothbrush for children',
    '174': 'Breeze Log Pure Gold 100x3',
    '175': 'Clean soft face150x3',
    '176': 'Banbu 100x3',
    '177': 'Vader Baby 150x3',
    '178': 'Xiangyin minions 150x3',
    '179': 'Breeze log pure black Yao 150x3',
    '18': 'Huiyi Dried Sweet Potato 228g',
    '180': 'Clean cloud velvet touch 130x3',
    '181': 'Shu Jie Meng Printing 120x2',
    '182': 'Xiangyin Hongyue 130x3',
    '183': 'Depot Applewood Flavor 90x4',
    '184': 'Breeze and toughness pure product 130x3',
    '185': 'Goldfish bamboo pulp green 135x3',
    '186': 'Breeze log pure product 150x2',
    '187': 'Clean soft face130x3',
    '188': 'Vida stereo beauty 110x3',
    '189': 'Clean Soft CS Single Pack*',
    '19': 'Huiyi Thai Dried Mango 80g',
    '190': 'Xiangyin Minions Single Pack*',
    '191': 'Fresh Breeze Single Pack*',
    '192': 'Xiangyin tea language package*',
    '193': 'Breeze texture pure product single bag*',
    '194': 'Mickey 1928 Notebook',
    '195': 'Guangbo Solid Glue 15g',
    '196': 'bill file bag',
    '197': 'Morning light snail correction tape',
    '198': 'Hongtai Liquid Glue 50g',
    '199': 'Maped self-adhesive labels',
    '2': 'Vegetable garden biscuits 80g',
    '20': 'Huiyi Dried Yellow Peach 75g',
    '200': 'east asian marker pen',
    '21': 'Huiyi Lemon Slices 65g',
    '22': 'Xinjiang Hetian Tan Jujube 454g',
    '23': 'Huiyi Shiitake Mushroom 100g',
    '24': 'Huiyi Dried Longan 500g',
    '25': 'Huiyi Tea Tree Mushroom 200g',
    '26': 'Haoxiong Black Fungus Single Slice 150g',
    '27': 'Huiyi Boiled Peanuts 454g',
    '28': 'Huiyi Day Lily 100g',
    '29': 'Chacha Herbal Tea Melon Seeds 150g',
    '3': 'Shanghaojia Shrimp Chips 40g',
    '30': 'Chacha Milk Flavor Melon Seeds 150g',
    '31': 'Chezai Tea Bag Green Tea 50g',
    '32': 'Chezai Tea Bag Black Tea 50g',
    '33': 'Premium sweet potato flavor 80g',
    '34': 'Youlemei Red Bean Milk Tea 65g',
    '35': 'Huanni Potato Porridge 25g',
    '36': 'Jiangzhong Hougu Breakfast Rice Thin 40g',
    '37': 'Yonghe Soy Milk Sweet Soy Milk Powder 210g',
    '38': 'Lipton Lemon Flavored Tea 180g',
    '39': 'Quaker Multi Berry Oatmeal 40g',
    '4': 'Shanghaojia Crab Flavor Yizu 40g',
    '40': 'Rongyigu Maijia Black Rice Flavor 30g',
    '41': 'Rongyigu Maijia Red Bean Flavor 30g',
    '42': 'Konno Spicy Beef Noodles 112g',
    '43': 'Jinye Laotan Sauerkraut Beef Noodles 118g',
    '44': 'Konno Braised Beef Noodles 114g',
    '45': 'Hewei Seafood Flavor 84g',
    '46': 'Master Kong White Pepper Meat Bone Noodles 76g',
    '47': 'Master Kong Spicy Beef Noodles 105g',
    '48': 'Master Kang Spicy Garlic Pork Rib Noodles 108g',
    '49': 'Master Kong Rattan Pepper Beef Noodles 82g',
    '5': 'Miao crispy magic charcoal flavor 65g',
    '50': 'Huafeng Chicken Three Fresh E-Noodles 87g',
    '51': 'Master Kong Black Pepper Steak Noodles 104g',
    '52': 'Wugudaochang Braised Beef Noodles 100g',
    '53': 'Master Kong Laotan Sauerkraut Beef Noodles 114g',
    '54': 'Aji Puff Cookies Mango Pineapple Flavor 60g',
    '55': 'Qinglian Blueberry Flavor Sandwich Cake 63g',
    '56': 'Qinglian Pineapple Flavor Sandwich Cake 63g',
    '57': 'Qinglian Strawberry Flavor Sandwich Cake 63g',
    '58': 'Garden Wafer Biscuits Strawberry Flavor 50g',
    '59': 'Garden Lemon Wafer Biscuits 50g',
    '6': 'Panpan BBQ Steak Flavor Cubes 105g',
    '60': 'Escher Vanilla Milk Flavor 50g',
    '61': 'Esther Chocolate Flavor 50g',
    '62': 'Hyakuriki Shigeru seaweed flavor 60g',
    '63': 'Pretz Strawberry Milk Flavor 45g',
    '64': 'Nestle Crispy Shark 80g',
    '65': 'Nabanti Chocolate Flavored Wafers 58g',
    '66': 'Guili Mediterranean Style Breadsticks 50g',
    '67': 'Master Kong Miaofu Chocolate Flavor 48g',
    '68': 'Love Folks Record Bread 90g',
    '69': 'Daliyuan Pie strawberry flavor single pack*',
    '7': 'Shanghaojia Shrimp Crackers 40g',
    '70': 'mini Oreo 55g',
    '71': 'Nongfu Mountain Spring Mineral Water 550ml',
    '72': "C'estbon Mineral Water 555ml",
    '73': 'Coca-Cola Zero 500ml',
    '74': 'Coca-Cola 500ml',
    '75': 'Pepsi 600ml',
    '76': 'Fanta Apple Flavor 500ml',
    '77': 'Fanta Orange 500ml',
    '78': 'Sprite 500ml',
    '79': 'Heineken Beer 500ml',
    '8': 'Shanghaojia Onion Rings 40g',
    '80': 'Budweiser 600ml',
    '81': 'Pepsi 330ml',
    '82': 'Coca-Cola 330ml',
    '83': 'Wanglaoji 310ml',
    '84': 'Chapai Yuzu Green Tea 500ml',
    '85': 'Cha Pai Rose Lychee Black Tea 500ml',
    '86': 'Master Kong Iced Black Tea 250ml',
    '87': 'Jiaduobao 250ml',
    '88': 'RIO Fruit Wine Peach Flavor 275ml',
    '89': 'RIO Fruit Wine Blue Rose Whiskey Flavor 275ml',
    '9': '50g',
    '90': 'Niulanshan Erguotou 100ml',
    '91': 'Harbin Beer 330ml',
    '92': 'Tsingtao Beer 330ml',
    '93': 'Snow Beer 330ml',
    '94': 'Harbin Beer 500ml',
    '95': 'KELER beer 500ml',
    '96': 'Budweiser 500ml',
    '97': 'QQ Star Quancong Milk 125ml',
    '98': 'QQ Star Food Milk 125ml',
    '99': 'Wahaha AD Calcium Milk 220g'
}

STRINGS = sorted([
    str(i) for i in range(1,201)
])

RESNET_PATH = 'weights/resnet_model_128_aug.h5'
RESNET_MODEL = tf.keras.models.load_model(RESNET_PATH)

def classify_image(
    img,
    img_height = 128,
    img_width = 128
):
    image = np.expand_dims(
        img.resize((img_width,img_height)),
        axis=0
    )
    category_id = STRINGS[np.argmax(
        RESNET_MODEL.predict(image)
    )]
    return STRINGS_TO_NAMES[category_id]



def convert_to_x1y1x2y2_format(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack((x1, y1, x2, y2), axis=1)


def convert_to_xywh_format(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack((x, y, w, h), axis=1)


def non_max_suppression_tf(boxes, scores, iou_threshold):
    selected_indices = tf.image.non_max_suppression(
        convert_to_x1y1x2y2_format(boxes),
        scores,
        max_output_size=100,
        iou_threshold=iou_threshold
    )
    return selected_indices


def save_bboxed_images(
        image_path,
        bbox_data,
        output_folder='cropped_images'
):
    img = plt.imread(image_path)

    os.makedirs(output_folder, exist_ok=True)

    grouped = bbox_data.groupby('label')

    for label, group in grouped:
        print(f"label is {label}, the type is {type(label)}\n\n")
        for idx, row in group.iterrows():
            w = row['w'] * img.shape[1]
            h = row['h'] * img.shape[0]
            x = row['x'] * img.shape[1] - w / 2
            y = row['y'] * img.shape[0] - h / 2

            # Crop and save the image
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cropped_img = img[y1:y2, x1:x2]
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            os.makedirs(os.path.join(output_folder, str(label)), exist_ok=True)
            output_path = os.path.join(output_folder, str(label), f'label_{label}_bbox_{idx}.png')
            cv2.imwrite(output_path, cropped_img)


def load_images_from_folder(folder_path, target_height=400):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # Calculate the target width while preserving the aspect ratio
        aspect_ratio = img.width / img.height
        target_width = int(target_height * aspect_ratio)

        # Resize the image
        resized_img = ImageOps.fit(img, (target_width, target_height), Image.ANTIALIAS)

        images.append(resized_img)
    return images


def show_cropped_images(
        base_folder
):
    st.title("Detected objects:")

    # base_folder = "images"
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    images = []
    for subfolder in subfolders:
        # st.subheader(subfolder)
        images += load_images_from_folder(subfolder)

    num_images = len(images)
    num_cols = min(num_images, 5)  # Limit the number of columns to 5
    num_rows = (num_images + num_cols - 1) // num_cols

    for row in range(num_rows):
        # cols = st.beta_columns(num_cols)
        cols = st.columns(num_cols)
        for col in range(num_cols):
            index = row * num_cols + col
            if index < num_images:
                description = classify_image(images[index])
                cols[col].image(images[index], use_column_width=True)
                cols[col].write(description)


st.title("Retail product object detection")

uploaded_file = st.file_uploader("Upload an image!", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded picture as 'uploaded_pic.jpg'
    unique_foldername = f"{str(uuid.uuid4())}"
    unique_filename = unique_foldername + "/uploaded_pic.jpg"

    # But first delete it if exists
    # for windows
    # process = subprocess.run(
    #     "rmdir /s /q " + unique_foldername,
    #     shell=True
    # )

    # for unix
    process = subprocess.run("rm -rf " + unique_foldername, shell=True)

    print(f"unique_foldername = {unique_foldername}")
    print(f"unique_filename = {unique_filename}")

    os.makedirs(unique_foldername, exist_ok=True)
    with open(unique_filename, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.write("Image uploaded successfully!")

    # Step 2: Run detection script
    st.write("Running object detection on the uploaded image...")

    command = "python detect.py"
    command_parameters = " --weights weights/weights_trained_by_sam.pt"
    command_parameters += " --source " + unique_filename
    command_parameters += " --conf 0.1 --no-trace --save-txt"
    command_parameters += " --save-conf  --project " + unique_foldername
    process = subprocess.Popen(
        command + command_parameters,
        shell=True,
        stdout=subprocess.PIPE
    )
    process.wait()

    st.write("Object detection completed!")

    result_image_path = unique_foldername + "/exp/uploaded_pic.jpg"

    # Step 3: Display resulting image
    if os.path.isfile(result_image_path):
        result_image = open(result_image_path, "rb").read()
        st.image(result_image, caption="YOLOv7 resulting Image", use_column_width=True)
    else:
        st.write("No resulting image found.")

    # Save cropped images.
    data = pd.read_csv(
        unique_foldername + "/exp/labels/uploaded_pic.txt",
        sep=' ',
        names=[
            'label',
            'x',
            'y',
            'w',
            'h',
            'conf'
        ]
    )

    boxes = data[['x', 'y', 'w', 'h']].to_numpy()
    scores = data['conf'].to_numpy()

    selected_indices = non_max_suppression_tf(
        boxes=boxes,
        scores=scores,
        iou_threshold=0.2
    )

    filtered_data = data.iloc[selected_indices].reset_index(drop=True)

    save_bboxed_images(
        image_path=unique_filename,
        bbox_data=filtered_data,
        output_folder=unique_foldername + '/cropped_images'
    )



    show_cropped_images(
        unique_foldername + '/cropped_images'
    )

    # end delete in the end
    # for windows
    # process = subprocess.run(
    #     "rmdir /s /q " + unique_foldername,
    #     shell=True
    # )

    # for unix
    process = subprocess.run("rm -rf " + unique_foldername, shell=True)

else:
    st.write("Please upload an image.")
