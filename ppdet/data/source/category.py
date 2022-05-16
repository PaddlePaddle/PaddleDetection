# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from ppdet.data.source.voc import pascalvoc_label
from ppdet.data.source.widerface import widerface_label
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['get_categories']


def get_categories(metric_type, anno_file=None, arch=None):
    """
    Get class id to category id map and category id
    to category name map from annotation file.

    Args:
        metric_type (str): metric type, currently support 'coco', 'voc', 'oid'
            and 'widerface'.
        anno_file (str): annotation file path
    """
    if arch == 'keypoint_arch':
        return (None, {'id': 'keypoint'})

    if anno_file == None or (not os.path.isfile(anno_file)):
        logger.warning(
            "anno_file '{}' is None or not set or not exist, "
            "please recheck TrainDataset/EvalDataset/TestDataset.anno_path, "
            "otherwise the default categories will be used by metric_type.".
            format(anno_file))

    if metric_type.lower() == 'coco' or metric_type.lower(
    ) == 'rbox' or metric_type.lower() == 'snipercoco':
        if anno_file and os.path.isfile(anno_file):
            if anno_file.endswith('json'):
                # lazy import pycocotools here
                from pycocotools.coco import COCO
                coco = COCO(anno_file)
                cats = coco.loadCats(coco.getCatIds())

                clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
                catid2name = {cat['id']: cat['name'] for cat in cats}

            elif anno_file.endswith('txt'):
                cats = []
                with open(anno_file) as f:
                    for line in f.readlines():
                        cats.append(line.strip())
                if cats[0] == 'background': cats = cats[1:]

                clsid2catid = {i: i for i in range(len(cats))}
                catid2name = {i: name for i, name in enumerate(cats)}

            else:
                raise ValueError("anno_file {} should be json or txt.".format(
                    anno_file))
            return clsid2catid, catid2name

        # anno file not exist, load default categories of COCO17
        else:
            if metric_type.lower() == 'rbox':
                logger.warning(
                    "metric_type: {}, load default categories of DOTA.".format(
                        metric_type))
                return _dota_category()
            logger.warning("metric_type: {}, load default categories of COCO.".
                           format(metric_type))
            return _coco17_category()

    elif metric_type.lower() == 'voc':
        if anno_file and os.path.isfile(anno_file):
            cats = []
            with open(anno_file) as f:
                for line in f.readlines():
                    cats.append(line.strip())

            if cats[0] == 'background':
                cats = cats[1:]

            clsid2catid = {i: i for i in range(len(cats))}
            catid2name = {i: name for i, name in enumerate(cats)}

            return clsid2catid, catid2name

        # anno file not exist, load default categories of
        # VOC all 20 categories
        else:
            logger.warning("metric_type: {}, load default categories of VOC.".
                           format(metric_type))
            return _vocall_category()

    elif metric_type.lower() == 'oid':
        if anno_file and os.path.isfile(anno_file):
            logger.warning("only default categories support for OID19")
        return _oid19_category()

    elif metric_type.lower() == 'widerface':
        return _widerface_category()

    elif metric_type.lower() == 'keypointtopdowncocoeval' or metric_type.lower(
    ) == 'keypointtopdownmpiieval':
        return (None, {'id': 'keypoint'})

    elif metric_type.lower() in ['mot', 'motdet', 'reid']:
        if anno_file and os.path.isfile(anno_file):
            cats = []
            with open(anno_file) as f:
                for line in f.readlines():
                    cats.append(line.strip())
            if cats[0] == 'background':
                cats = cats[1:]
            clsid2catid = {i: i for i in range(len(cats))}
            catid2name = {i: name for i, name in enumerate(cats)}
            return clsid2catid, catid2name
        # anno file not exist, load default category 'pedestrian'.
        else:
            logger.warning(
                "metric_type: {}, load default categories of pedestrian MOT.".
                format(metric_type))
            return _mot_category(category='pedestrian')

    elif metric_type.lower() in ['kitti', 'bdd100kmot']:
        return _mot_category(category='vehicle')

    elif metric_type.lower() in ['mcmot']:
        if anno_file and os.path.isfile(anno_file):
            cats = []
            with open(anno_file) as f:
                for line in f.readlines():
                    cats.append(line.strip())
            if cats[0] == 'background':
                cats = cats[1:]
            clsid2catid = {i: i for i in range(len(cats))}
            catid2name = {i: name for i, name in enumerate(cats)}
            return clsid2catid, catid2name
        # anno file not exist, load default categories of visdrone all 10 categories
        else:
            logger.warning(
                "metric_type: {}, load default categories of VisDrone.".format(
                    metric_type))
            return _visdrone_category()

    else:
        raise ValueError("unknown metric type {}".format(metric_type))


def _mot_category(category='pedestrian'):
    """
    Get class id to category id map and category id
    to category name map of mot dataset
    """
    label_map = {category: 0}
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def _coco17_category():
    """
    Get class id to category id map and category id
    to category name map of COCO2017 dataset

    """
    clsid2catid = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: 24,
        24: 25,
        25: 27,
        26: 28,
        27: 31,
        28: 32,
        29: 33,
        30: 34,
        31: 35,
        32: 36,
        33: 37,
        34: 38,
        35: 39,
        36: 40,
        37: 41,
        38: 42,
        39: 43,
        40: 44,
        41: 46,
        42: 47,
        43: 48,
        44: 49,
        45: 50,
        46: 51,
        47: 52,
        48: 53,
        49: 54,
        50: 55,
        51: 56,
        52: 57,
        53: 58,
        54: 59,
        55: 60,
        56: 61,
        57: 62,
        58: 63,
        59: 64,
        60: 65,
        61: 67,
        62: 70,
        63: 72,
        64: 73,
        65: 74,
        66: 75,
        67: 76,
        68: 77,
        69: 78,
        70: 79,
        71: 80,
        72: 81,
        73: 82,
        74: 84,
        75: 85,
        76: 86,
        77: 87,
        78: 88,
        79: 89,
        80: 90
    }

    catid2name = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }

    clsid2catid = {k - 1: v for k, v in clsid2catid.items()}
    catid2name.pop(0)

    return clsid2catid, catid2name


def _dota_category():
    """
    Get class id to category id map and category id
    to category name map of dota dataset
    """
    catid2name = {
        0: 'background',
        1: 'plane',
        2: 'baseball-diamond',
        3: 'bridge',
        4: 'ground-track-field',
        5: 'small-vehicle',
        6: 'large-vehicle',
        7: 'ship',
        8: 'tennis-court',
        9: 'basketball-court',
        10: 'storage-tank',
        11: 'soccer-ball-field',
        12: 'roundabout',
        13: 'harbor',
        14: 'swimming-pool',
        15: 'helicopter'
    }
    catid2name.pop(0)
    clsid2catid = {i: i + 1 for i in range(len(catid2name))}
    return clsid2catid, catid2name


def _vocall_category():
    """
    Get class id to category id map and category id
    to category name map of mixup voc dataset

    """
    label_map = pascalvoc_label()
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def _widerface_category():
    label_map = widerface_label()
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]
    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def _oid19_category():
    clsid2catid = {k: k + 1 for k in range(500)}

    catid2name = {
        0: "background",
        1: "Infant bed",
        2: "Rose",
        3: "Flag",
        4: "Flashlight",
        5: "Sea turtle",
        6: "Camera",
        7: "Animal",
        8: "Glove",
        9: "Crocodile",
        10: "Cattle",
        11: "House",
        12: "Guacamole",
        13: "Penguin",
        14: "Vehicle registration plate",
        15: "Bench",
        16: "Ladybug",
        17: "Human nose",
        18: "Watermelon",
        19: "Flute",
        20: "Butterfly",
        21: "Washing machine",
        22: "Raccoon",
        23: "Segway",
        24: "Taco",
        25: "Jellyfish",
        26: "Cake",
        27: "Pen",
        28: "Cannon",
        29: "Bread",
        30: "Tree",
        31: "Shellfish",
        32: "Bed",
        33: "Hamster",
        34: "Hat",
        35: "Toaster",
        36: "Sombrero",
        37: "Tiara",
        38: "Bowl",
        39: "Dragonfly",
        40: "Moths and butterflies",
        41: "Antelope",
        42: "Vegetable",
        43: "Torch",
        44: "Building",
        45: "Power plugs and sockets",
        46: "Blender",
        47: "Billiard table",
        48: "Cutting board",
        49: "Bronze sculpture",
        50: "Turtle",
        51: "Broccoli",
        52: "Tiger",
        53: "Mirror",
        54: "Bear",
        55: "Zucchini",
        56: "Dress",
        57: "Volleyball",
        58: "Guitar",
        59: "Reptile",
        60: "Golf cart",
        61: "Tart",
        62: "Fedora",
        63: "Carnivore",
        64: "Car",
        65: "Lighthouse",
        66: "Coffeemaker",
        67: "Food processor",
        68: "Truck",
        69: "Bookcase",
        70: "Surfboard",
        71: "Footwear",
        72: "Bench",
        73: "Necklace",
        74: "Flower",
        75: "Radish",
        76: "Marine mammal",
        77: "Frying pan",
        78: "Tap",
        79: "Peach",
        80: "Knife",
        81: "Handbag",
        82: "Laptop",
        83: "Tent",
        84: "Ambulance",
        85: "Christmas tree",
        86: "Eagle",
        87: "Limousine",
        88: "Kitchen & dining room table",
        89: "Polar bear",
        90: "Tower",
        91: "Football",
        92: "Willow",
        93: "Human head",
        94: "Stop sign",
        95: "Banana",
        96: "Mixer",
        97: "Binoculars",
        98: "Dessert",
        99: "Bee",
        100: "Chair",
        101: "Wood-burning stove",
        102: "Flowerpot",
        103: "Beaker",
        104: "Oyster",
        105: "Woodpecker",
        106: "Harp",
        107: "Bathtub",
        108: "Wall clock",
        109: "Sports uniform",
        110: "Rhinoceros",
        111: "Beehive",
        112: "Cupboard",
        113: "Chicken",
        114: "Man",
        115: "Blue jay",
        116: "Cucumber",
        117: "Balloon",
        118: "Kite",
        119: "Fireplace",
        120: "Lantern",
        121: "Missile",
        122: "Book",
        123: "Spoon",
        124: "Grapefruit",
        125: "Squirrel",
        126: "Orange",
        127: "Coat",
        128: "Punching bag",
        129: "Zebra",
        130: "Billboard",
        131: "Bicycle",
        132: "Door handle",
        133: "Mechanical fan",
        134: "Ring binder",
        135: "Table",
        136: "Parrot",
        137: "Sock",
        138: "Vase",
        139: "Weapon",
        140: "Shotgun",
        141: "Glasses",
        142: "Seahorse",
        143: "Belt",
        144: "Watercraft",
        145: "Window",
        146: "Giraffe",
        147: "Lion",
        148: "Tire",
        149: "Vehicle",
        150: "Canoe",
        151: "Tie",
        152: "Shelf",
        153: "Picture frame",
        154: "Printer",
        155: "Human leg",
        156: "Boat",
        157: "Slow cooker",
        158: "Croissant",
        159: "Candle",
        160: "Pancake",
        161: "Pillow",
        162: "Coin",
        163: "Stretcher",
        164: "Sandal",
        165: "Woman",
        166: "Stairs",
        167: "Harpsichord",
        168: "Stool",
        169: "Bus",
        170: "Suitcase",
        171: "Human mouth",
        172: "Juice",
        173: "Skull",
        174: "Door",
        175: "Violin",
        176: "Chopsticks",
        177: "Digital clock",
        178: "Sunflower",
        179: "Leopard",
        180: "Bell pepper",
        181: "Harbor seal",
        182: "Snake",
        183: "Sewing machine",
        184: "Goose",
        185: "Helicopter",
        186: "Seat belt",
        187: "Coffee cup",
        188: "Microwave oven",
        189: "Hot dog",
        190: "Countertop",
        191: "Serving tray",
        192: "Dog bed",
        193: "Beer",
        194: "Sunglasses",
        195: "Golf ball",
        196: "Waffle",
        197: "Palm tree",
        198: "Trumpet",
        199: "Ruler",
        200: "Helmet",
        201: "Ladder",
        202: "Office building",
        203: "Tablet computer",
        204: "Toilet paper",
        205: "Pomegranate",
        206: "Skirt",
        207: "Gas stove",
        208: "Cookie",
        209: "Cart",
        210: "Raven",
        211: "Egg",
        212: "Burrito",
        213: "Goat",
        214: "Kitchen knife",
        215: "Skateboard",
        216: "Salt and pepper shakers",
        217: "Lynx",
        218: "Boot",
        219: "Platter",
        220: "Ski",
        221: "Swimwear",
        222: "Swimming pool",
        223: "Drinking straw",
        224: "Wrench",
        225: "Drum",
        226: "Ant",
        227: "Human ear",
        228: "Headphones",
        229: "Fountain",
        230: "Bird",
        231: "Jeans",
        232: "Television",
        233: "Crab",
        234: "Microphone",
        235: "Home appliance",
        236: "Snowplow",
        237: "Beetle",
        238: "Artichoke",
        239: "Jet ski",
        240: "Stationary bicycle",
        241: "Human hair",
        242: "Brown bear",
        243: "Starfish",
        244: "Fork",
        245: "Lobster",
        246: "Corded phone",
        247: "Drink",
        248: "Saucer",
        249: "Carrot",
        250: "Insect",
        251: "Clock",
        252: "Castle",
        253: "Tennis racket",
        254: "Ceiling fan",
        255: "Asparagus",
        256: "Jaguar",
        257: "Musical instrument",
        258: "Train",
        259: "Cat",
        260: "Rifle",
        261: "Dumbbell",
        262: "Mobile phone",
        263: "Taxi",
        264: "Shower",
        265: "Pitcher",
        266: "Lemon",
        267: "Invertebrate",
        268: "Turkey",
        269: "High heels",
        270: "Bust",
        271: "Elephant",
        272: "Scarf",
        273: "Barrel",
        274: "Trombone",
        275: "Pumpkin",
        276: "Box",
        277: "Tomato",
        278: "Frog",
        279: "Bidet",
        280: "Human face",
        281: "Houseplant",
        282: "Van",
        283: "Shark",
        284: "Ice cream",
        285: "Swim cap",
        286: "Falcon",
        287: "Ostrich",
        288: "Handgun",
        289: "Whiteboard",
        290: "Lizard",
        291: "Pasta",
        292: "Snowmobile",
        293: "Light bulb",
        294: "Window blind",
        295: "Muffin",
        296: "Pretzel",
        297: "Computer monitor",
        298: "Horn",
        299: "Furniture",
        300: "Sandwich",
        301: "Fox",
        302: "Convenience store",
        303: "Fish",
        304: "Fruit",
        305: "Earrings",
        306: "Curtain",
        307: "Grape",
        308: "Sofa bed",
        309: "Horse",
        310: "Luggage and bags",
        311: "Desk",
        312: "Crutch",
        313: "Bicycle helmet",
        314: "Tick",
        315: "Airplane",
        316: "Canary",
        317: "Spatula",
        318: "Watch",
        319: "Lily",
        320: "Kitchen appliance",
        321: "Filing cabinet",
        322: "Aircraft",
        323: "Cake stand",
        324: "Candy",
        325: "Sink",
        326: "Mouse",
        327: "Wine",
        328: "Wheelchair",
        329: "Goldfish",
        330: "Refrigerator",
        331: "French fries",
        332: "Drawer",
        333: "Treadmill",
        334: "Picnic basket",
        335: "Dice",
        336: "Cabbage",
        337: "Football helmet",
        338: "Pig",
        339: "Person",
        340: "Shorts",
        341: "Gondola",
        342: "Honeycomb",
        343: "Doughnut",
        344: "Chest of drawers",
        345: "Land vehicle",
        346: "Bat",
        347: "Monkey",
        348: "Dagger",
        349: "Tableware",
        350: "Human foot",
        351: "Mug",
        352: "Alarm clock",
        353: "Pressure cooker",
        354: "Human hand",
        355: "Tortoise",
        356: "Baseball glove",
        357: "Sword",
        358: "Pear",
        359: "Miniskirt",
        360: "Traffic sign",
        361: "Girl",
        362: "Roller skates",
        363: "Dinosaur",
        364: "Porch",
        365: "Human beard",
        366: "Submarine sandwich",
        367: "Screwdriver",
        368: "Strawberry",
        369: "Wine glass",
        370: "Seafood",
        371: "Racket",
        372: "Wheel",
        373: "Sea lion",
        374: "Toy",
        375: "Tea",
        376: "Tennis ball",
        377: "Waste container",
        378: "Mule",
        379: "Cricket ball",
        380: "Pineapple",
        381: "Coconut",
        382: "Doll",
        383: "Coffee table",
        384: "Snowman",
        385: "Lavender",
        386: "Shrimp",
        387: "Maple",
        388: "Cowboy hat",
        389: "Goggles",
        390: "Rugby ball",
        391: "Caterpillar",
        392: "Poster",
        393: "Rocket",
        394: "Organ",
        395: "Saxophone",
        396: "Traffic light",
        397: "Cocktail",
        398: "Plastic bag",
        399: "Squash",
        400: "Mushroom",
        401: "Hamburger",
        402: "Light switch",
        403: "Parachute",
        404: "Teddy bear",
        405: "Winter melon",
        406: "Deer",
        407: "Musical keyboard",
        408: "Plumbing fixture",
        409: "Scoreboard",
        410: "Baseball bat",
        411: "Envelope",
        412: "Adhesive tape",
        413: "Briefcase",
        414: "Paddle",
        415: "Bow and arrow",
        416: "Telephone",
        417: "Sheep",
        418: "Jacket",
        419: "Boy",
        420: "Pizza",
        421: "Otter",
        422: "Office supplies",
        423: "Couch",
        424: "Cello",
        425: "Bull",
        426: "Camel",
        427: "Ball",
        428: "Duck",
        429: "Whale",
        430: "Shirt",
        431: "Tank",
        432: "Motorcycle",
        433: "Accordion",
        434: "Owl",
        435: "Porcupine",
        436: "Sun hat",
        437: "Nail",
        438: "Scissors",
        439: "Swan",
        440: "Lamp",
        441: "Crown",
        442: "Piano",
        443: "Sculpture",
        444: "Cheetah",
        445: "Oboe",
        446: "Tin can",
        447: "Mango",
        448: "Tripod",
        449: "Oven",
        450: "Mouse",
        451: "Barge",
        452: "Coffee",
        453: "Snowboard",
        454: "Common fig",
        455: "Salad",
        456: "Marine invertebrates",
        457: "Umbrella",
        458: "Kangaroo",
        459: "Human arm",
        460: "Measuring cup",
        461: "Snail",
        462: "Loveseat",
        463: "Suit",
        464: "Teapot",
        465: "Bottle",
        466: "Alpaca",
        467: "Kettle",
        468: "Trousers",
        469: "Popcorn",
        470: "Centipede",
        471: "Spider",
        472: "Sparrow",
        473: "Plate",
        474: "Bagel",
        475: "Personal care",
        476: "Apple",
        477: "Brassiere",
        478: "Bathroom cabinet",
        479: "studio couch",
        480: "Computer keyboard",
        481: "Table tennis racket",
        482: "Sushi",
        483: "Cabinetry",
        484: "Street light",
        485: "Towel",
        486: "Nightstand",
        487: "Rabbit",
        488: "Dolphin",
        489: "Dog",
        490: "Jug",
        491: "Wok",
        492: "Fire hydrant",
        493: "Human eye",
        494: "Skyscraper",
        495: "Backpack",
        496: "Potato",
        497: "Paper towel",
        498: "Lifejacket",
        499: "Bicycle wheel",
        500: "Toilet",
    }

    return clsid2catid, catid2name


def _visdrone_category():
    clsid2catid = {i: i for i in range(10)}

    catid2name = {
        0: 'pedestrian',
        1: 'people',
        2: 'bicycle',
        3: 'car',
        4: 'van',
        5: 'truck',
        6: 'tricycle',
        7: 'awning-tricycle',
        8: 'bus',
        9: 'motor'
    }
    return clsid2catid, catid2name
