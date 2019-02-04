import os
import sys
import json

image_class = [
                "3m_high_tack_spray_adhesive",
                "aunt_jemima_original_syrup",
                "campbells_chicken_noodle_soup",
                "cheez_it_white_cheddar",
                "cholula_chipotle_hot_sauce",
                "clif_crunch_chocolate_chip",
                "coca_cola_glass_bottle",
                "detergent",
                "expo_marker_red",
                "listerine_green",
                "nice_honey_roasted_almonds",
                "nutrigrain_apple_cinnamon",
                "palmolive_green",
                "pringles_bbq",
                "vo5_extra_body_volumizing_shampoo",
                "vo5_split_ends_anti_breakage_shampoo"
                ]

image_pos = ['1', '2']

test_images = [
    "test_1",
    "test_2",
    "test_3",
    "easy_single_1",
    "easy_single_2",
    "easy_single_3",
    "hard_single_1",
    "hard_single_2",
    "hard_single_3",
    "easy_multi_1",
    "easy_multi_2",
    "easy_multi_3",
    "hard_multi_1",
    "hard_multi_2",
    "hard_multi_3"
]

mask1 = [[250,75],[375,290]]
mask2 = [[225,60],[360,240]]

img_db = {}
train_db = []
test_db = []
for ic in image_class:
    for ip in image_pos:
        for n in range(500):
            img = {}
            if os.path.isfile('train/'+ic+'/N'+ip+'_'+str(n)+'.jpg'):
                img['name'] = 'N'+ip+'_'+str(n)
                img['class'] = ic
                img['path'] = 'train/'+ic+'/N'+ip+'_'+str(n)+'.jpg'
                if ip == '1':
                    img['corners']=mask1
                else:
                    img['corners']=mask2
                train_db.append(img)

for ti in test_images:
    img = {}
    if os.path.isfile('sample_test/'+ti+'.jpg'):
        img['name'] = ti
        img['path'] = 'sample_test/'+ti+'.jpg'
        test_db.append(img)

img_db['train'] = train_db
img_db['test'] = test_db
with open('image_description.json', 'w') as f:
    json.dump(img_db, f)