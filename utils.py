
from __future__ import print_function
import os
import wandb
from sklearn.metrics import confusion_matrix 
import numpy as np
from collections import  Counter
from collections import OrderedDict
from tqdm import tqdm
import copy
from collections import defaultdict
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.Linear5 import Linear5
from models.LeNet5 import LeNet5
from models.inceptionresnetv2 import inceptionresnetv2
from models.vit import vit_b_16, vit_b_32,  vit_l_16, vit_l_32, vit_h_14
np.set_printoptions(suppress=True)

    
def get_dataset(args):
    if "fmnist" in args.dataset.lower():
        train_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.2860,), (0.3530,)),
            ])

        test_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.2860,), (0.3530,))
            ])
        if args.dataset.lower() =="fmnist":
            if args.train_transform:
                dataset1 = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.FashionMNIST(args.data_path, train=False, transform=test_transform)
            args.num_classes = 10
            args.class_label_names = [i for i in range(10)]
        else:
            raise ValueError 
    elif "mnist" in args.dataset.lower():
        train_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,)),
            ])
        
        test_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,))
            ])
        if args.dataset.lower() =="mnist":
            if args.train_transform:
                dataset1 = datasets.MNIST(args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.MNIST(args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.MNIST(args.data_path, train=False, transform=test_transform)
            args.num_classes = 10
            args.class_label_names = [i for i in range(10)]
        else:
            raise ValueError 
    elif "cifar" in args.dataset.lower():
        train_transform=v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32, 4),
            v2.ToTensor(),
            v2.Normalize(mean=[0.491, 0.482, 0.447],
                                        std=[0.247, 0.243, 0.262]),      
            ])
        
        test_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.491, 0.482, 0.447],
                                        std=[0.247, 0.243, 0.262]),   
            ])
        if args.dataset.lower() == "cifar10":
            if args.train_transform:
                dataset1 = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
            args.num_classes=10
            args.class_label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        elif args.dataset.lower() == "cifar100":
            if args.train_transform:
                dataset1 = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)
            args.num_classes=100
            args.class_label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        else:
            raise ValueError
        
    elif "imagenet" in args.dataset.lower() or "webvision" in args.dataset.lower():
        if args.arch.lower() == "inceptionresnetv2":
            args.resize_image = 299
        else:
            args.resize_image = 224
        train_transform=v2.Compose([ 
            v2.RandomResizedCrop((args.resize_image,args.resize_image)),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),    
            ])

        test_transform=v2.Compose([
            v2.Resize((args.resize_image,args.resize_image)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])     
            ])
        if args.dataset.lower() == "imagenette":
            if args.train_transform:
                dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "imagenette","train"), transform=train_transform)
            else:
                dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "imagenette","train"), transform=test_transform)
            dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "imagenette","val"), transform=test_transform)
            args.num_classes = 10
            args.class_label_names = ["bench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
        else:
            args.num_classes = 1000
            args.class_label_names = ['tench, Tinca tinca', 'goldfish, Carassius auratus', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'tiger shark, Galeocerdo cuvieri', 'hammerhead, hammerhead shark', 'electric ray, crampfish, numbfish, torpedo', 'stingray', 'cock', 'hen', 'ostrich, Struthio camelus', 'brambling, Fringilla montifringilla', 'goldfinch, Carduelis carduelis', 'house finch, linnet, Carpodacus mexicanus', 'junco, snowbird', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'robin, American robin, Turdus migratorius', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel, dipper', 'kite', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture', 'great grey owl, great gray owl, Strix nebulosa', 'European fire salamander, Salamandra salamandra', 'common newt, Triturus vulgaris', 'eft', 'spotted salamander, Ambystoma maculatum', 'axolotl, mud puppy, Ambystoma mexicanum', 'bullfrog, Rana catesbeiana', 'tree frog, tree-frog', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 'loggerhead, loggerhead turtle, Caretta caretta', 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'mud turtle', 'terrapin', 'box turtle, box tortoise', 'banded gecko', 'common iguana, iguana, Iguana iguana', 'American chameleon, anole, Anolis carolinensis', 'whiptail, whiptail lizard', 'agama', 'frilled lizard, Chlamydosaurus kingi', 'alligator lizard', 'Gila monster, Heloderma suspectum', 'green lizard, Lacerta viridis', 'African chameleon, Chamaeleo chamaeleon', 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 'African crocodile, Nile crocodile, Crocodylus niloticus', 'American alligator, Alligator mississipiensis', 'triceratops', 'thunder snake, worm snake, Carphophis amoenus', 'ringneck snake, ring-necked snake, ring snake', 'hognose snake, puff adder, sand viper', 'green snake, grass snake', 'king snake, kingsnake', 'garter snake, grass snake', 'water snake', 'vine snake', 'night snake, Hypsiglena torquata', 'boa constrictor, Constrictor constrictor', 'rock python, rock snake, Python sebae', 'Indian cobra, Naja naja', 'green mamba', 'sea snake', 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'sidewinder, horned rattlesnake, Crotalus cerastes', 'trilobite', 'harvestman, daddy longlegs, Phalangium opilio', 'scorpion', 'black and gold garden spider, Argiope aurantia', 'barn spider, Araneus cavaticus', 'garden spider, Aranea diademata', 'black widow, Latrodectus mactans', 'tarantula', 'wolf spider, hunting spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse, partridge, Bonasa umbellus', 'prairie chicken, prairie grouse, prairie fowl', 'peacock', 'quail', 'partridge', 'African grey, African gray, Psittacus erithacus', 'macaw', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser, Mergus serrator', 'goose', 'black swan, Cygnus atratus', 'tusker', 'echidna, spiny anteater, anteater', 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus', 'wallaby, brush kangaroo', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'wombat', 'jellyfish', 'sea anemone, anemone', 'brain coral', 'flatworm, platyhelminth', 'nematode, nematode worm, roundworm', 'conch', 'snail', 'slug', 'sea slug, nudibranch', 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 'chambered nautilus, pearly nautilus, nautilus', 'Dungeness crab, Cancer magister', 'rock crab, Cancer irroratus', 'fiddler crab', 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'crayfish, crawfish, crawdad, crawdaddy', 'hermit crab', 'isopod', 'white stork, Ciconia ciconia', 'black stork, Ciconia nigra', 'spoonbill', 'flamingo', 'little blue heron, Egretta caerulea', 'American egret, great white heron, Egretta albus', 'bittern', 'crane', 'limpkin, Aramus pictus', 'European gallinule, Porphyrio porphyrio', 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'bustard', 'ruddy turnstone, Arenaria interpres', 'red-backed sandpiper, dunlin, Erolia alpina', 'redshank, Tringa totanus', 'dowitcher', 'oystercatcher, oyster catcher', 'pelican', 'king penguin, Aptenodytes patagonica', 'albatross, mollymawk', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'dugong, Dugong dugon', 'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog, Maltese terrier, Maltese', 'Pekinese, Pekingese, Peke', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound, Afghan', 'basset, basset hound', 'beagle', 'bloodhound, sleuthhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound, Walker foxhound', 'English foxhound', 'redbone', 'borzoi, Russian wolfhound', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound, Ibizan Podenco', 'Norwegian elkhound, elkhound', 'otterhound, otter hound', 'Saluki, gazelle hound', 'Scottish deerhound, deerhound', 'Weimaraner', 'Staffordshire bullterrier, Staffordshire bull terrier', 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier, Sealyham', 'Airedale, Airedale terrier', 'cairn, cairn terrier', 'Australian terrier', 'Dandie Dinmont, Dandie Dinmont terrier', 'Boston bull, Boston terrier', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier, Scottish terrier, Scottie', 'Tibetan terrier, chrysanthemum dog', 'silky terrier, Sydney silky', 'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa, Lhasa apso', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla, Hungarian pointer', 'English setter', 'Irish setter, red setter', 'Gordon setter', 'Brittany spaniel', 'clumber, clumber spaniel', 'English springer, English springer spaniel', 'Welsh springer spaniel', 'cocker spaniel, English cocker spaniel, cocker', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog, bobtail', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'collie', 'Border collie', 'Bouvier des Flandres, Bouviers des Flandres', 'Rottweiler', 'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard, St Bernard', 'Eskimo dog, husky', 'malamute, malemute, Alaskan malamute', 'Siberian husky', 'dalmatian, coach dog, carriage dog', 'affenpinscher, monkey pinscher, monkey dog', 'basenji', 'pug, pug-dog', 'Leonberg', 'Newfoundland, Newfoundland dog', 'Great Pyrenees', 'Samoyed, Samoyede', 'Pomeranian', 'chow, chow chow', 'keeshond', 'Brabancon griffon', 'Pembroke, Pembroke Welsh corgi', 'Cardigan, Cardigan Welsh corgi', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf, grey wolf, gray wolf, Canis lupus', 'white wolf, Arctic wolf, Canis lupus tundrarum', 'red wolf, maned wolf, Canis rufus, Canis niger', 'coyote, prairie wolf, brush wolf, Canis latrans', 'dingo, warrigal, warragal, Canis dingo', 'dhole, Cuon alpinus', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'hyena, hyaena', 'red fox, Vulpes vulpes', 'kit fox, Vulpes macrotis', 'Arctic fox, white fox, Alopex lagopus', 'grey fox, gray fox, Urocyon cinereoargenteus', 'tabby, tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat, Siamese', 'Egyptian cat', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'lynx, catamount', 'leopard, Panthera pardus', 'snow leopard, ounce, Panthera uncia', 'jaguar, panther, Panthera onca, Felis onca', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'brown bear, bruin, Ursus arctos', 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'sloth bear, Melursus ursinus, Ursus ursinus', 'mongoose', 'meerkat, mierkat', 'tiger beetle', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'ground beetle, carabid beetle', 'long-horned beetle, longicorn, longicorn beetle', 'leaf beetle, chrysomelid', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'cricket', 'walking stick, walkingstick, stick insect', 'cockroach, roach', 'mantis, mantid', 'cicada, cicala', 'leafhopper', 'lacewing, lacewing fly', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'damselfly', 'admiral', 'ringlet, ringlet butterfly', 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cabbage butterfly', 'sulphur butterfly, sulfur butterfly', 'lycaenid, lycaenid butterfly', 'starfish, sea star', 'sea urchin', 'sea cucumber, holothurian', 'wood rabbit, cottontail, cottontail rabbit', 'hare', 'Angora, Angora rabbit', 'hamster', 'porcupine, hedgehog', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'marmot', 'beaver', 'guinea pig, Cavia cobaya', 'sorrel', 'zebra', 'hog, pig, grunter, squealer, Sus scrofa', 'wild boar, boar, Sus scrofa', 'warthog', 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'ox', 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 'bison', 'ram, tup', 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 'ibex, Capra ibex', 'hartebeest', 'impala, Aepyceros melampus', 'gazelle', 'Arabian camel, dromedary, Camelus dromedarius', 'llama', 'weasel', 'mink', 'polecat, fitch, foulmart, foumart, Mustela putorius', 'black-footed ferret, ferret, Mustela nigripes', 'otter', 'skunk, polecat, wood pussy', 'badger', 'armadillo', 'three-toed sloth, ai, Bradypus tridactylus', 'orangutan, orang, orangutang, Pongo pygmaeus', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'gibbon, Hylobates lar', 'siamang, Hylobates syndactylus, Symphalangus syndactylus', 'guenon, guenon monkey', 'patas, hussar monkey, Erythrocebus patas', 'baboon', 'macaque', 'langur', 'colobus, colobus monkey', 'proboscis monkey, Nasalis larvatus', 'marmoset', 'capuchin, ringtail, Cebus capucinus', 'howler monkey, howler', 'titi, titi monkey', 'spider monkey, Ateles geoffroyi', 'squirrel monkey, Saimiri sciureus', 'Madagascar cat, ring-tailed lemur, Lemur catta', 'indri, indris, Indri indri, Indri brevicaudatus', 'Indian elephant, Elephas maximus', 'African elephant, Loxodonta africana', 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 'barracouta, snoek', 'eel', 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch', 'rock beauty, Holocanthus tricolor', 'anemone fish', 'sturgeon', 'gar, garfish, garpike, billfish, Lepisosteus osseus', 'lionfish', 'puffer, pufferfish, blowfish, globefish', 'abacus', 'abaya', "academic gown, academic robe, judge's robe", 'accordion, piano accordion, squeeze box', 'acoustic guitar', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'airliner', 'airship, dirigible', 'altar', 'ambulance', 'amphibian, amphibious vehicle', 'analog clock', 'apiary, bee house', 'apron', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'bakery, bakeshop, bakehouse', 'balance beam, beam', 'balloon', 'ballpoint, ballpoint pen, ballpen, Biro', 'Band Aid', 'banjo', 'bannister, banister, balustrade, balusters, handrail', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel, cask', 'barrow, garden cart, lawn cart, wheelbarrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap, swimming cap', 'bath towel', 'bathtub, bathing tub, bath, tub', 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'beacon, lighthouse, beacon light, pharos', 'beaker', 'bearskin, busby, shako', 'beer bottle', 'beer glass', 'bell cote, bell cot', 'bib', 'bicycle-built-for-two, tandem bicycle, tandem', 'bikini, two-piece', 'binder, ring-binder', 'binoculars, field glasses, opera glasses', 'birdhouse', 'boathouse', 'bobsled, bobsleigh, bob', 'bolo tie, bolo, bola tie, bola', 'bonnet, poke bonnet', 'bookcase', 'bookshop, bookstore, bookstall', 'bottlecap', 'bow', 'bow tie, bow-tie, bowtie', 'brass, memorial tablet, plaque', 'brassiere, bra, bandeau', 'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'breastplate, aegis, egis', 'broom', 'bucket, pail', 'buckle', 'bulletproof vest', 'bullet train, bullet', 'butcher shop, meat market', 'cab, hack, taxi, taxicab', 'caldron, cauldron', 'candle, taper, wax light', 'cannon', 'canoe', 'can opener, tin opener', 'cardigan', 'car mirror', 'carousel, carrousel, merry-go-round, roundabout, whirligig', "carpenter's kit, tool kit", 'carton', 'car wheel', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello, violoncello', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'chain', 'chainlink fence', 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 'chain saw, chainsaw', 'chest', 'chiffonier, commode', 'chime, bell, gong', 'china cabinet, china closet', 'Christmas stocking', 'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace', 'cleaver, meat cleaver, chopper', 'cliff dwelling', 'cloak', 'clog, geta, patten, sabot', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil, spiral, volute, whorl, helix', 'combination lock', 'computer keyboard, keypad', 'confectionery, confectionary, candy store', 'container ship, containership, container vessel', 'convertible', 'corkscrew, bottle screw', 'cornet, horn, trumpet, trump', 'cowboy boot', 'cowboy hat, ten-gallon hat', 'cradle', 'crane', 'crash helmet', 'crate', 'crib, cot', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam, dike, dyke', 'desk', 'desktop computer', 'dial telephone, dial phone', 'diaper, nappy, napkin', 'digital clock', 'digital watch', 'dining table, board', 'dishrag, dishcloth', 'dishwasher, dish washer, dishwashing machine', 'disk brake, disc brake', 'dock, dockage, docking facility', 'dogsled, dog sled, dog sleigh', 'dome', 'doormat, welcome mat', 'drilling platform, offshore rig', 'drum, membranophone, tympan', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan, blower', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso maker', 'face powder', 'feather boa, boa', 'file, file cabinet, filing cabinet', 'fireboat', 'fire engine, fire truck', 'fire screen, fireguard', 'flagpole, flagstaff', 'flute, transverse flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster', 'freight car', 'French horn, horn', 'frying pan, frypan, skillet', 'fur coat', 'garbage truck, dustcart', 'gasmask, respirator, gas helmet', 'gas pump, gasoline pump, petrol pump, island dispenser', 'goblet', 'go-kart', 'golf ball', 'golfcart, golf cart', 'gondola', 'gong, tam-tam', 'gown', 'grand piano, grand', 'greenhouse, nursery, glasshouse', 'grille, radiator grille', 'grocery store, grocery, food market, market', 'guillotine', 'hair slide', 'hair spray', 'half track', 'hammer', 'hamper', 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'hand-held computer, hand-held microcomputer', 'handkerchief, hankie, hanky, hankey', 'hard disc, hard disk, fixed disk', 'harmonica, mouth organ, harp, mouth harp', 'harp', 'harvester, reaper', 'hatchet', 'holster', 'home theater, home theatre', 'honeycomb', 'hook, claw', 'hoopskirt, crinoline', 'horizontal bar, high bar', 'horse cart, horse-cart', 'hourglass', 'iPod', 'iron, smoothing iron', "jack-o'-lantern", 'jean, blue jean, denim', 'jeep, landrover', 'jersey, T-shirt, tee shirt', 'jigsaw puzzle', 'jinrikisha, ricksha, rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat, laboratory coat', 'ladle', 'lampshade, lamp shade', 'laptop, laptop computer', 'lawn mower, mower', 'lens cap, lens cover', 'letter opener, paper knife, paperknife', 'library', 'lifeboat', 'lighter, light, igniter, ignitor', 'limousine, limo', 'liner, ocean liner', 'lipstick, lip rouge', 'Loafer', 'lotion', 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', "loupe, jeweler's loupe", 'lumbermill, sawmill', 'magnetic compass', 'mailbag, postbag', 'mailbox, letter box', 'maillot', 'maillot, tank suit', 'manhole cover', 'maraca', 'marimba, xylophone', 'mask', 'matchstick', 'maypole', 'maze, labyrinth', 'measuring cup', 'medicine chest, medicine cabinet', 'megalith, megalithic structure', 'microphone, mike', 'microwave, microwave oven', 'military uniform', 'milk can', 'minibus', 'miniskirt, mini', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home, manufactured home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net', 'motor scooter, scooter', 'mountain bike, all-terrain bike, off-roader', 'mountain tent', 'mouse, computer mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook, notebook computer', 'obelisk', 'oboe, hautboy, hautbois', 'ocarina, sweet potato', 'odometer, hodometer, mileometer, milometer', 'oil filter', 'organ, pipe organ', 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle, boat paddle', 'paddlewheel, paddle wheel', 'padlock', 'paintbrush', "pajama, pyjama, pj's, jammies", 'palace', 'panpipe, pandean pipe, syrinx', 'paper towel', 'parachute, chute', 'parallel bars, bars', 'park bench', 'parking meter', 'passenger car, coach, carriage', 'patio, terrace', 'pay-phone, pay-station', 'pedestal, plinth, footstall', 'pencil box, pencil case', 'pencil sharpener', 'perfume, essence', 'Petri dish', 'photocopier', 'pick, plectrum, plectron', 'pickelhaube', 'picket fence, paling', 'pickup, pickup truck', 'pier', 'piggy bank, penny bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate, pirate ship', 'pitcher, ewer', "plane, carpenter's plane, woodworking plane", 'planetarium', 'plastic bag', 'plate rack', 'plow, plough', "plunger, plumber's helper", 'Polaroid camera, Polaroid Land camera', 'pole', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'poncho', 'pool table, billiard table, snooker table', 'pop bottle, soda bottle', 'pot, flowerpot', "potter's wheel", 'power drill', 'prayer rug, prayer mat', 'printer', 'prison, prison house', 'projectile, missile', 'projector', 'puck, hockey puck', 'punching bag, punch bag, punching ball, punchball', 'purse', 'quill, quill pen', 'quilt, comforter, comfort, puff', 'racer, race car, racing car', 'racket, racquet', 'radiator', 'radio, wireless', 'radio telescope, radio reflector', 'rain barrel', 'recreational vehicle, RV, R.V.', 'reel', 'reflex camera', 'refrigerator, icebox', 'remote control, remote', 'restaurant, eating house, eating place, eatery', 'revolver, six-gun, six-shooter', 'rifle', 'rocking chair, rocker', 'rotisserie', 'rubber eraser, rubber, pencil eraser', 'rugby ball', 'rule, ruler', 'running shoe', 'safe', 'safety pin', 'saltshaker, salt shaker', 'sandal', 'sarong', 'sax, saxophone', 'scabbard', 'scale, weighing machine', 'school bus', 'schooner', 'scoreboard', 'screen, CRT screen', 'screw', 'screwdriver', 'seat belt, seatbelt', 'sewing machine', 'shield, buckler', 'shoe shop, shoe-shop, shoe store', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule, slipstick', 'sliding door', 'slot, one-armed bandit', 'snorkel', 'snowmobile', 'snowplow, snowplough', 'soap dispenser', 'soccer ball', 'sock', 'solar dish, solar collector, solar furnace', 'sombrero', 'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula', 'speedboat', "spider web, spider's web", 'spindle', 'sports car, sport car', 'spotlight, spot', 'stage', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch, stop watch', 'stove', 'strainer', 'streetcar, tram, tramcar, trolley, trolley car', 'stretcher', 'studio couch, day bed', 'stupa, tope', 'submarine, pigboat, sub, U-boat', 'suit, suit of clothes', 'sundial', 'sunglass', 'sunglasses, dark glasses, shades', 'sunscreen, sunblock, sun blocker', 'suspension bridge', 'swab, swob, mop', 'sweatshirt', 'swimming trunks, bathing trunks', 'swing', 'switch, electric switch, electrical switch', 'syringe', 'table lamp', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'tape player', 'teapot', 'teddy, teddy bear', 'television, television system', 'tennis ball', 'thatch, thatched roof', 'theater curtain, theatre curtain', 'thimble', 'thresher, thrasher, threshing machine', 'throne', 'tile roof', 'toaster', 'tobacco shop, tobacconist shop, tobacconist', 'toilet seat', 'torch', 'totem pole', 'tow truck, tow car, wrecker', 'toyshop', 'tractor', 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'tray', 'trench coat', 'tricycle, trike, velocipede', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus, trolley coach, trackless trolley', 'trombone', 'tub, vat', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle, monocycle', 'upright, upright piano', 'vacuum, vacuum cleaner', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin, fiddle', 'volleyball', 'waffle iron', 'wall clock', 'wallet, billfold, notecase, pocketbook', 'wardrobe, closet, press', 'warplane, military plane', 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'washer, automatic washer, washing machine', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool, woolen, woollen', 'worm fence, snake fence, snake-rail fence, Virginia fence', 'wreck', 'yawl', 'yurt', 'web site, website, internet site, site', 'comic book', 'crossword puzzle, crossword', 'street sign', 'traffic light, traffic signal, stoplight', 'book jacket, dust cover, dust jacket, dust wrapper', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot, hotpot', 'trifle', 'ice cream, icecream', 'ice lolly, lolly, lollipop, popsicle', 'French loaf', 'bagel, beigel', 'pretzel', 'cheeseburger', 'hotdog, hot dog, red hot', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini, courgette', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber, cuke', 'artichoke, globe artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple, ananas', 'banana', 'jackfruit, jak, jack', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate sauce, chocolate syrup', 'dough', 'meat loaf, meatloaf', 'pizza, pizza pie', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff, drop, drop-off', 'coral reef', 'geyser', 'lakeside, lakeshore', 'promontory, headland, head, foreland', 'sandbar, sand bar', 'seashore, coast, seacoast, sea-coast', 'valley, vale', 'volcano', 'ballplayer, baseball player', 'groom, bridegroom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'corn', 'acorn', 'hip, rose hip, rosehip', 'buckeye, horse chestnut, conker', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn, carrion fungus', 'earthstar', 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa', 'bolete', 'ear, spike, capitulum', 'toilet tissue, toilet paper, bathroom tissue']
            if args.dataset.lower() == "imagenet1k" or args.dataset.lower() == "imagenet":
                if args.train_transform:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=train_transform)
                else:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=test_transform)
                dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=test_transform)
            elif  "webvision" in args.dataset.lower() and "mini" in args.dataset.lower():
                args.num_classes = 50
                args.class_label_names = args.class_label_names[:50]
                if args.train_transform:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=train_transform)
                else:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=test_transform)
                dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=test_transform)
            elif "webvision" in args.dataset.lower():
                if args.train_transform:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=train_transform)
                else:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=test_transform)
                dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=test_transform)
            else:
                raise ValueError
    elif "clothing" in args.dataset.lower():
        if args.arch.lower() == "inceptionresnetv2":
            args.resize_image = 299
        else:
            args.resize_image = 224

        train_transform=v2.Compose([ 
            v2.RandomResizedCrop((args.resize_image,args.resize_image)),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),    
            ])

        test_transform=v2.Compose([
            v2.Resize((args.resize_image,args.resize_image)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])     
            ])
        args.num_classes = 14
        args.class_label_names = ["T-Shirt", "Shirt", "Knitwear", "Chiffon", "Sweater", "Hoodie", "Windbreaker", "Jacket", "Downcoat", "Suit", "Shawl", "Dress", "Vest", "Underwear"]
        if args.train_transform:
            dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "noisy_train"), transform=train_transform)
        else:
            dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "noisy_train"), transform=test_transform)
        dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=test_transform)
    else:
        raise ValueError
    return dataset1, dataset2

def get_test_set_clothing(args):
    if args.arch.lower() == "inceptionresnetv2":
        args.resize_image = 299
    else:
        args.resize_image = 224

    test_transform=v2.Compose([
        v2.Resize((args.resize_image,args.resize_image)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])     
        ])
    test_set = datasets.ImageFolder(root=os.path.join(args.data_path, "test"), transform=test_transform)
    return test_set

def get_mislabeled_dataset(dataset, percentage_mislabeled, num_classes, return_clean_partition, path):
    if os.path.exists(f"{path}.mislabeled_points"):
        # load the mislabeled targets
        (index_list, old_targets, updated_targets) = torch.load(f"{path}.mislabeled_points") 
    else:
        if percentage_mislabeled == 0:
            return dataset, None, (None, None, None)
        # Generate and save the mislabeled targets
        num_of_data_points = len(dataset)
        num_of_mislabeled = int(percentage_mislabeled * num_of_data_points)
        r=np.arange(num_of_data_points)
        np.random.shuffle(r)
        index_list = r[:num_of_mislabeled].tolist()
        updated_targets = []
        old_targets = []
        for sample_index in index_list:
            if (torch.is_tensor(dataset.targets[sample_index])):
                old_targets.append(dataset.targets[sample_index].item())
                updated_targets.append(random.choice([val for val in range(num_classes) if val != dataset.targets[sample_index].item()]))
            elif isinstance(dataset.targets[sample_index], int ):
                old_targets.append(dataset.targets[sample_index])
                updated_targets.append(random.choice([val for val in range(num_classes) if val != dataset.targets[sample_index]]))            
        
        torch.save((index_list, old_targets, updated_targets), f"{path}.mislabeled_points")
    #Update the targets
    corrupt_samples = []
    for sample_number, sample_index in enumerate(index_list):
        if (torch.is_tensor(dataset.targets[sample_index])):
            dataset.targets[sample_index] = torch.ones_like(dataset.targets[0]) * updated_targets[sample_number]
            corrupt_samples.append(dataset[sample_index][0])
        elif isinstance(dataset.targets[sample_index], int ):
            dataset.targets[sample_index] = int(updated_targets[sample_number] )
            corrupt_samples.append(dataset[sample_index][0])

    if corrupt_samples:
        corrupt_samples = torch.stack(corrupt_samples)
        if len(corrupt_samples.shape)<4:
            corrupt_samples = corrupt_samples.unsqueeze(1)
    if return_clean_partition:
        mask = np.ones(len(dataset), dtype=bool)
        mask[index_list] = False
        dataset.data  = dataset.data[mask]
        dataset.targets = torch.tensor(dataset.targets)[mask]
    return dataset, corrupt_samples, (index_list, old_targets, updated_targets)

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.v_label = torch.tensor ([0 for _ in range(len(dataset))]).long()
        self.indices = torch.tensor ([i for i in range(len(dataset))]).long()

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, self.v_label[index], self.indices[index]

    def __len__(self):
        return len(self.dataset)

def get_model(args,device):
    if  ( ("linear" in args.arch.lower() or "lenet" in args.arch.lower()) and ("mnist" not  in args.dataset.lower() ) ) or \
    ( ("vgg" in args.arch.lower() or "resnet" in args.arch.lower()) and ("mnist" in args.dataset.lower()) ) \
    or ("vit" in args.arch.lower() and  not ( "imagenet" in args.dataset.lower() or "webvision" in args.dataset.lower() or "clothing" in args.dataset.lower() ) ):
        # line 1 -> Linear or LeNet arch -> MNIST Dataset
        # line 2 -> VGG or ResNet arch -> Not MNIST Dataset
        # line 3 -> VIT ->  imagenet or webvision or clothing dataset
        raise ValueError
    # Instantiate model
    if "linear" in args.arch.lower():
        model = Linear5(bias="nobias" not in args.arch.lower()).to(device)
    elif "lenet" in args.arch.lower():
        model = LeNet5(bias="nobias" not in args.arch.lower()).to(device)
    elif "vgg11_bn" in args.arch.lower():
        model = vgg11_bn(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg13_bn" in args.arch.lower():
        model = vgg13_bn(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg16_bn" in args.arch.lower():
        model = vgg16_bn(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg19_bn" in args.arch.lower():
        model = vgg19_bn(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg11" in args.arch.lower():
        model = vgg11(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg13" in args.arch.lower():
        model = vgg13(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg16" in args.arch.lower():
        model = vgg16(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg19" in args.arch.lower():
        model = vgg19(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet152" in args.arch.lower():
        model = ResNet152(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet101" in args.arch.lower():
        model = ResNet101(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet50" in args.arch.lower():
        model = ResNet50(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet34" in args.arch.lower():
        model = ResNet34(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet18" in args.arch.lower():
        model = ResNet18(num_classes=args.num_classes,dataset=args.dataset.lower()).to(device)  
    elif "inceptionresnetv2" in args.arch.lower():
        model = inceptionresnetv2(num_classes=args.num_classes, pretrained=False).to(device)
    elif "vit_b_16" in args.arch.lower():
        model = vit_b_16().to(device)
    elif "vit_b_32" in args.arch.lower():
        model = vit_b_32().to(device)
    elif "vit_l_16" in args.arch.lower():
        model = vit_l_16().to(device)
    elif "vit_l_32" in args.arch.lower():
        model = vit_l_32().to(device)
    elif "vit_h_14" in args.arch.lower():
        model = vit_h_14().to(device)
    else:
        raise ValueError
    return model


# SAM optimizer taken from https://github.com/davda54/sam/blob/main/sam.py
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# GLS loss function taken from https://github.com/UCSC-REAL/negative-label-smoothing
def loss_gls(logits, labels, smooth_rate=0.1):
    # logits: model prediction logits before the soft-max, with size [batch_size, classes]
    # labels: the (noisy) labels for evaluation, with size [batch_size]
    # smooth_rate: could go either positive or negative, 
    # smooth_rate candidates we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smooth_rate * smooth_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    return torch.sum(loss)/num_batch


# Early stopper taken from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# taken from https://github.com/LJY-HY/MentorMix_pytorch/blob/master/train_MentorNet.py
def MentorMixLoss(args, device, MentorNet, StudentNet, x_i, y_i,v_true, loss_p_prev, loss_p_second_prev, epoch):
    '''
    v_true is set to 0s in this version.
    inputs : 
        x_i         [bsz,C,H,W]
        outputs_i   [bsz,num_class]
        y_i         [bsz]
    intermediate :
        x_j         [bsz,C,H,W]
        outputs_j   [bsz,num_class]
        y_j         [bsz]
    outputs:
        loss        [float]
        gamma       [float]

    Simple threshold function is used as MentorNet in this repository.
    '''
    XLoss = torch.nn.CrossEntropyLoss(reduction='none')
    # MentorNet 1
    bsz = x_i.shape[0]
    x_i, y_i,v_true = x_i.to(device), y_i.to(device), v_true.to(device)
    with torch.no_grad():
        outputs_i = StudentNet(x_i) 
        loss = XLoss(outputs_i,y_i)                      
        loss_p = args.mnet_ema*loss_p_prev + (1-args.mnet_ema)*sorted(loss)[int(bsz*args.mnet_gamma_p)]
        loss_diff = loss-loss_p
        v = MentorNet(v_true,args.epochs, epoch,loss,loss_diff)   

        # Burn-in Process( Do only for few epoch)
        if epoch < min(args.mnet_burnin, args.epochs*0.2):
            v = torch.bernoulli(torch.ones_like(loss_diff)/2).to(device)

    P_v = cat.Categorical(F.softmax(v,dim=0))           
    indices_j = P_v.sample(y_i.shape)                   
    
    # Prepare Mixup
    x_j = x_i[indices_j]
    y_j = y_i[indices_j]
    
    # MIXUP
    Beta = diri.Dirichlet(torch.tensor([args.mmix_alpha for _ in range(2)]))
    lambdas = Beta.sample(y_i.shape).to(device)
    lambdas_max = lambdas.max(dim=1)[0]                 
    lambdas = v*lambdas_max + (1-v)*(1-lambdas_max)     
    x_tilde = x_i * lambdas.view(lambdas.size(0),1,1,1) + x_j * (1-lambdas).view(lambdas.size(0),1,1,1)
    outputs_tilde = StudentNet(x_tilde)
    
    # Second Reweight
    with torch.no_grad():
        loss = lambdas*XLoss(outputs_tilde,y_i) + (1-lambdas)*XLoss(outputs_tilde,y_j)
        loss_p_second = args.mnet_ema*loss_p_second_prev + (1-args.mnet_ema)*sorted(loss)[int(bsz*args.mnet_gamma_p)]
        loss_diff = loss-loss_p_second
        v_mix = MentorNet(v_true,args.epochs, epoch,loss,loss_diff)

        # Burn-in Process( Do only for few epoch)
        if epoch < min(args.mnet_burnin, args.epochs*0.2):
            v_mix = torch.bernoulli(torch.ones_like(loss_diff)/2).to(device)

    loss = lambdas*XLoss(outputs_tilde,y_i) + (1-lambdas)*XLoss(outputs_tilde,y_j)
    masked_loss = loss*v_mix
  
    return XLoss(outputs_i,y_i).mean(), masked_loss.mean(), loss_p, loss_p_second, v

# taken from https://github.com/LJY-HY/MentorMix_pytorch/blob/master/train_MentorNet.py
def MentorNetLoss(args, device, MentorNet, StudentNet, x_i, y_i,v_true, loss_p_prev,  epoch):
    '''
    v_true is set to 0s in this version.
    inputs : 
        x_i         [bsz,C,H,W]
        outputs_i   [bsz,num_class]
        y_i         [bsz]
    intermediate :
        x_j         [bsz,C,H,W]
        outputs_j   [bsz,num_class]
        y_j         [bsz]
    outputs:
        loss        [float]
        gamma       [float]

    Simple threshold function is used as MentorNet in this repository.
    '''
    XLoss = torch.nn.CrossEntropyLoss(reduction='none')
    # MentorNet 1
    bsz = x_i.shape[0]
    x_i, y_i,v_true = x_i.to(device), y_i.to(device), v_true.to(device)
    with torch.no_grad():
        outputs_i = StudentNet(x_i) 
        loss = XLoss(outputs_i,y_i)                      
        loss_p = args.mnet_ema*loss_p_prev + (1-args.mnet_ema)*sorted(loss)[int(bsz*args.mnet_gamma_p)]
        loss_diff = loss-loss_p
        v = MentorNet(v_true,args.epochs, epoch,loss,loss_diff)           
        
        # Burn-in Process( Do only for few epoch)
        if epoch < min(args.mnet_burnin, args.epochs*0.2):
            v = torch.bernoulli(torch.ones_like(loss_diff)/2).to(device)

    output = StudentNet(x_i)
    loss = XLoss(output,y_i)
    masked_loss = loss*v
    return loss.mean(), masked_loss.mean(), loss_p, v



### VERIFIX - Helper functions
def get_representation_matrix_mislabeled(net, device, data_loader, sample_indexs = [], num_classes = 10, prev_recur_proj_mat = None,
                               samples_per_set=1000, max_batch_size=150, max_samples=50000, set_name = "Clean Set"): 
    if sample_indexs: 
        # sort data per class and collect samples from required class.
        dataset = data_loader.dataset 
        samples_list = []
        r=np.arange(len(sample_indexs))
        np.random.shuffle(r)
        sample_indexs = np.array(sample_indexs)[r[:samples_per_set]].tolist()
        for index in sample_indexs:
            samples_list.append(dataset[index][0])
        sample_tensor = torch.stack(samples_list, 0).to(device)
        if len(sample_tensor.shape)<4:
            sample_tensor = sample_tensor.unsqueeze(1)
            print(sample_tensor.shape)
        # Gets prepresentations as dict of dicts # form dataloader without transform
        activations = None 
        net.eval()
        for batch in tqdm(torch.split(sample_tensor, max_batch_size, dim=0), desc=f"Extracting representation for {set_name}"):
            try:
                batch_activations = net.get_activations(batch, prev_recur_proj_mat)
            except:
                batch_activations = net.module.get_activations(batch, prev_recur_proj_mat)
            ### Instantinously compress the batch
            for loc in batch_activations.keys():
                for key in batch_activations[loc].keys():
                    if batch_activations[loc][key].shape[0]> (int(max_samples/(sample_tensor.shape[0]/max_batch_size)) +1):
                        ### Shuffle and return a subset of patches
                        r=np.arange(batch_activations[loc][key].shape[0])
                        np.random.shuffle(r)
                        b = r[:(int(max_samples/(sample_tensor.shape[0]/max_batch_size)) +1)]
                        batch_activations[loc][key] = batch_activations[loc][key][b].copy()
            ### Concatinate the samples
            if activations:
                for loc in batch_activations.keys():
                    for key in batch_activations[loc].keys():
                        activations[loc][key] = np.concatenate([activations[loc][key],batch_activations[loc][key]], 0)
            else:
                activations = batch_activations
        ### Final check for reducing the sample size
        sampled_activations={"pre":OrderedDict(),"post":OrderedDict(),}
        for loc in batch_activations.keys():
            for key in batch_activations[loc].keys():
                if activations[loc][key].shape[0]> max_samples:
                    ### Shuffle and return a subset of patches
                    r=np.arange(activations[loc][key].shape[0])
                    np.random.shuffle(r)
                    b = r[:max_samples]
                    sampled_activations[loc][key] = activations[loc][key][b].copy()
                else:
                    sampled_activations[loc][key] = activations[loc][key].copy()
        # Transpose activations
        loc_keys = list(sampled_activations.keys())
        act_keys =list(sampled_activations[loc_keys[0]].keys())
        mat_dict={loc:OrderedDict() for loc in loc_keys}
        for loc in loc_keys:
            for act in list(sampled_activations[loc].keys()):
                activation = sampled_activations[loc][act].transpose()
                mat_dict[loc][act]= activation
        #Prints the representation shapes.
        for loc in loc_keys:
            print('-'*30)
            print(f'Representation Matrix {loc} Layer for {set_name}')
            print('-'*30)    
            for act in list(sampled_activations[loc].keys()):
                print (f' Layer {act} : [{mat_dict[loc][act].shape}]')
            print('-'*30)
        return mat_dict
    else:
        return {loc:OrderedDict() for loc in ["pre", "post"]}
   
def get_SVD (mat_dict, set_name = "SVD"):
    feature_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    s_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    for loc in mat_dict.keys():
        for act in tqdm(mat_dict[loc].keys(), desc=f"{loc}layer - SVD for {set_name}"):
            activation = torch.Tensor(mat_dict[loc][act]).to("cuda")
            U,S,Vh = torch.linalg.svd(activation, full_matrices=False)
            U = U.cpu().numpy()
            S = S.cpu().numpy()            
            feature_dict[loc][act] = U
            s_dict[loc][act] = S
    return feature_dict,  s_dict

def select_basis(feature_dict, full_s_dict, threshold):
    if threshold is None:
        return feature_dict
    out_feature_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    for loc in feature_dict.keys():
        for act in feature_dict[loc].keys():
            U = feature_dict[loc][act]
            S = full_s_dict[loc][act]
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold) +1  
            out_feature_dict[loc][act] = U[:,:r]
    print('-'*40)
    print(f'Gradient Constraints Summary')
    print('-'*40)
    for loc in out_feature_dict.keys():
        for act in out_feature_dict[loc].keys():
            print (f'{loc} layer {act} : {out_feature_dict[loc][act].shape[1]}/{out_feature_dict[loc][act].shape[0]}')
    print('-'*40)
    return out_feature_dict


     
def get_scaled_feature_mat(feature_dict, full_s_dict, mode, alpha, device):
    feature_mat_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    # Projection Matrix Precomputation
    for loc in feature_dict.keys():
        for act in feature_dict[loc].keys():
            U = torch.Tensor( feature_dict[loc][act] ).to(device)
            S = full_s_dict[loc][act]
            r = U.shape[1]
            if mode == "baseline":
                importance = torch.ones(r).to(device) 
            elif mode == "gpm":
                importance = torch.ones(r).to(device) 
            elif mode == "sgp":
                importance = torch.Tensor(( alpha*S/( (alpha-1)*S+max(S)) )[:r]).to(device) 
            elif mode == "sap":
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                importance =  torch.Tensor(( alpha*sval_ratio/((alpha-1)*sval_ratio+1) ) [:r]).to(device) 
            else:
                raise ValueError
            U.requires_grad = False
            feature_mat_dict[loc][act] = torch.mm( U, torch.diag(importance**0.5) )
    return feature_mat_dict

def get_projections(feature_mat_retain_dict, feature_mat_unlearn_dict,projection_type, device):
    feature_mat = {"pre":OrderedDict(), "post":OrderedDict()}
    for loc in feature_mat_retain_dict.keys():
        for act in feature_mat_retain_dict[loc].keys():
            Ur = feature_mat_retain_dict[loc][act]
            Uf = feature_mat_unlearn_dict[loc][act]  
            Mr = torch.mm(Ur, Ur.transpose(0,1))     
            Mf = torch.mm(Uf, Uf.transpose(0,1))
            I = torch.eye(Mf.shape[0]).to(device) 
            Mri = torch.mm(Mr, Mf) # Intersection in terms of retain space basis
            Mfi = torch.mm(Mf, Mr)  # Intersection in terms of forget space basis
            # Select type of projection. 
            if projection_type == "baseline":
                feature_mat[loc][act]= I 
            elif projection_type == "Mr":
                feature_mat[loc][act]= Mr  
            elif projection_type == "I-Mf":
                feature_mat[loc][act]= I - Mf   
            elif projection_type == "Mr-Mi":
                feature_mat[loc][act]= Mr - Mri                             
            elif projection_type == "I-(Mf-Mi)":
                feature_mat[loc][act]= I - (Mf - Mfi)
            else:
                raise ValueError
    return feature_mat

def metric_function(x, y):
    out= x *(1 - y)
    return out
    
def test(model, device, data_loader, class_label_names, num_classes =10, \
                    plot_cm=False, verbose=True, set_name = "Val Set"):
    if data_loader is None or data_loader == []:
        return 0, 0
    model.eval()
    sample_loss = 0
    correct = 0
    cm = np.zeros((num_classes,num_classes))
    dict_classwise_acc={}
    dict_classwise_loss=defaultdict(float)
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc=set_name):
            data, target = data.to(device), target.to(device)
            output = model(data)
            sample_loss = F.cross_entropy(output, target, reduction='none')
            for i in range(num_classes):
                dict_classwise_loss[i] +=  torch.where(target == i, sample_loss, torch.zeros_like(sample_loss)).sum()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            cm+=confusion_matrix(target.cpu().numpy(),pred.squeeze(-1).cpu().numpy(), labels=[val for val in range(num_classes)])
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss = sum(list(dict_classwise_loss.values()))
    total_loss /= len(data_loader.dataset)    
    classwise_acc = cm.diagonal()/cm.sum(axis=1)
    for i in range(0,num_classes):
        dict_classwise_acc[class_label_names[i]] =  100*classwise_acc[i]
   
    if plot_cm:
        pass 
    print(f'{set_name}: Average loss: {total_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({(100. * correct / len(data_loader.dataset)):.0f}%)')
    if verbose:
        if num_classes <11:
            print('-'*30)
            print(f"{set_name} Confusion Matrix \n{cm}")
            print('-'*30)  
            print(f"{set_name} Class Wise ACC \n{dict_classwise_acc}")
        print('-'*30) 
        wandb.log({ 
                    f"{set_name}/acc": (100. * correct / len(data_loader.dataset)),
                    f"{set_name}/loss":  total_loss,
                    f"{set_name}/class-acc/test_acc":dict_classwise_acc,
                    },
                    step = 0
                    )
    return (100. * correct / len(data_loader.dataset)), total_loss

def get_regularized_curvature_for_batch(net, device, batch_data, batch_labels, h=1e-3, niter=10, temp=1):
        num_samples = batch_data.shape[0]
        net.eval()
        regr = torch.zeros(num_samples)
        eigs = torch.zeros(num_samples)
        for _ in range(niter):
            v = torch.randint_like(batch_data, high=2).to(device)
            # Generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = h * (v + 1e-7)
            batch_data.requires_grad_()
            outputs_pos = net(batch_data + v)
            outputs_orig = net(batch_data)
            loss_pos = F.cross_entropy(outputs_pos / temp, batch_labels)
            loss_orig = F.cross_entropy(outputs_orig / temp, batch_labels)
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), batch_data )[0]

            regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()
            eigs += torch.diag(torch.matmul(v.reshape(num_samples,-1), grad_diff.reshape(num_samples,-1).T)).cpu().detach()
            net.zero_grad()
            if batch_data.grad is not None:
                batch_data.grad.zero_()
        curv_estimate = eigs / niter
        regr_estimate = regr / niter
        return curv_estimate, regr_estimate

def get_high_low_curve_dataloader(args, dataset,  net, device, set_name="Train Set"):
    cache_path = os.path.join(args.load_loc, f"{args.dataset}_{args.arch}_MisLabeled{args.percentage_mislabeled}_seed{args.seed}.classwise_highlowcur_index")
    if os.path.exists(cache_path):
        print(f"Loaded from cached file at {cache_path}" )
        classwise_indices = torch.load(cache_path )       
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=16,shuffle = False)
        curvature = []
        targets = []

        for images, labels in tqdm(dataloader, desc=set_name ):
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True
            net.zero_grad()
            curvature.append( get_regularized_curvature_for_batch(net, device, images, labels)[1].detach().clone().cpu() )
            targets.append(labels.detach().clone().cpu() )
        _, indices = torch.sort(torch.cat(curvature), descending=True)
        targets = torch.cat(targets).tolist()
        # To ensure class balanced highcurvature/lowcurvature dataloader.
        classwise_indices = [[] for i in range(args.num_classes)] 
        for index in indices.tolist():
            classwise_indices[targets[index]].append(index)
        torch.save( classwise_indices , cache_path) 

    highcur_samples_per_class = (args.forget_samples//args.num_classes+1)
    lowcur_samples_per_class = (args.retain_samples//args.num_classes+1)
    highcur_index = []
    lowcur_index = []
    for i in range(args.num_classes):
        highcur_index.extend(classwise_indices[i][:highcur_samples_per_class])
        if args.percentile_low_curve>=0:
            low_start_index = int(args.percentile_low_curve * len(classwise_indices[i]))
            lowcur_index.extend(classwise_indices[i][len(classwise_indices[i])-lowcur_samples_per_class-low_start_index:len(classwise_indices[i])-low_start_index])
        else:
            low_start_index = int(-1*args.percentile_low_curve * len(classwise_indices[i]))
            lowcur_index.extend(random.sample(classwise_indices[i][len(classwise_indices[i])-low_start_index:len(classwise_indices[i])], lowcur_samples_per_class ))

    highcur_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, highcur_index), batch_size=args.batch_size, pin_memory=True, num_workers=16,shuffle = False)
    lowcur_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, lowcur_index), batch_size=args.batch_size, pin_memory=True, num_workers=16,shuffle = False)

    return highcur_dataloader, lowcur_dataloader


def activation_projection_based_unlearning(args, model, 
                                           train_loaders, 
                                           val_loaders, 
                                           test_loader, 
                                           trainset,
                                           device, 
                                           round = 0, 
                                           prev_recur_proj_mat = None):
    train_loader, train_loader_clean, train_loader_corrupt = train_loaders
    val_loader, val_loader_clean, val_loader_corrupt = val_loaders
    if round==0:
        run = wandb.init(
                        # Set the project where this run will be logged
                        project=f"Verifix-{args.dataset}-{args.project_name}",
                        group= f"report-{args.projection_location[-1]}layer-{args.group_name}", 
                        name=f"{args.run_name}_initial_test_acc",
                        entity=args.entity_name,
                        dir = os.environ["LOCAL_HOME"],
                        # Track hyperparameters and run metadata
                        config= vars(args)
            )  
        inference_model = copy.deepcopy(model)
    else:
        inference_model = copy.deepcopy(model)
        try:
            inference_model.module.project_weights(prev_recur_proj_mat, args.project_classifier)
        except:
            inference_model.project_weights(prev_recur_proj_mat, args.project_classifier)
    # Initial Evaluation
    corrupt_acc, clean_loss = test(inference_model, device, train_loader_corrupt, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Corrupt Train Set", verbose=prev_recur_proj_mat is None)
    if args.use_valset:
        full_acc, full_loss = test(inference_model, device, val_loader, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Full Val Set", verbose=prev_recur_proj_mat is None)
        base_metric= full_acc  
        if val_loader_clean:
            clean_acc, clean_loss = test(inference_model, device, val_loader_clean, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Clean Val Set", verbose=prev_recur_proj_mat is None)
            base_metric= clean_acc 
        
    else:
        full_acc, full_loss = test(inference_model, device, train_loader, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Full Train Set", verbose=prev_recur_proj_mat is None)
        base_metric = full_acc 
        if train_loader_clean:
            clean_acc, clean_loss = test(inference_model, device, train_loader_clean, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Clean Train Set", verbose=prev_recur_proj_mat is None)
            base_metric = clean_acc    
        
        
        
    test_acc, test_loss = test(inference_model, device, test_loader, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Test Set", verbose=prev_recur_proj_mat is None)
    if prev_recur_proj_mat is None:
        wandb.finish() 

    # Gets the basis vectors for retain set    
    if args.use_curvature:
        highcur_dataloader, lowcur_dataloader = get_high_low_curve_dataloader(args, trainset,  model, device, set_name="Train Dataset")
        mat_retain_dict = get_representation_matrix_mislabeled(model, device, lowcur_dataloader, sample_indexs= np.arange(len(lowcur_dataloader.dataset)).tolist(), num_classes = args.num_classes, 
                                                            prev_recur_proj_mat=prev_recur_proj_mat, samples_per_set=args.retain_samples, 
                                                            max_batch_size=args.max_batch_size, max_samples=args.max_samples, set_name = "Clean Set")  
        full_feature_retain_dict, full_s_retain_dict = get_SVD( mat_retain_dict, f"SVD Clean Set") 

                                                                
        # Gets the basis vectors for Forget set    
        mat_unlearn_dict= get_representation_matrix_mislabeled(model, device, highcur_dataloader,  sample_indexs= np.arange(len(highcur_dataloader.dataset)).tolist(), num_classes = args.num_classes, 
                                                            prev_recur_proj_mat=prev_recur_proj_mat, samples_per_set=args.forget_samples, 
                                                            max_batch_size=args.max_batch_size, max_samples=args.max_samples, set_name = "Corrupt Set") 
        full_feature_unlearn_dict, full_s_unlearn_dict= get_SVD( mat_unlearn_dict, f"SVD Corrupt Set") 

    else:
        mat_retain_dict = get_representation_matrix_mislabeled(model, device, train_loader_clean, sample_indexs= np.arange(len(train_loader_clean.dataset)).tolist(), num_classes = args.num_classes, 
                                                            prev_recur_proj_mat=prev_recur_proj_mat, samples_per_set=args.retain_samples, 
                                                            max_batch_size=args.max_batch_size, max_samples=args.max_samples, set_name = "Clean Set")  
        full_feature_retain_dict, full_s_retain_dict = get_SVD( mat_retain_dict, f"SVD Clean Set") 

                                                                
        # Gets the basis vectors for Forget set    
        mat_unlearn_dict= get_representation_matrix_mislabeled(model, device, train_loader_corrupt,  sample_indexs= np.arange(len(train_loader_corrupt.dataset)).tolist(), num_classes = args.num_classes, 
                                                            prev_recur_proj_mat=prev_recur_proj_mat, samples_per_set=args.forget_samples, 
                                                            max_batch_size=args.max_batch_size, max_samples=args.max_samples, set_name = "Corrupt Set") 
        full_feature_unlearn_dict, full_s_unlearn_dict= get_SVD( mat_unlearn_dict, f"SVD Corrupt Set") 

    best_metric = base_metric
    next_recur_proj_mat = None
    unlearnt_model = copy.deepcopy(model)
    num_layer = len(full_s_retain_dict["pre"])
    for mode in args.mode:
        # Iterate over all the retain modes
        for mode_forget in args.mode_forget:
            # Iterate over all the forget modes
            for projection_type in args.projection_type:
                # Iterate over all the projection types
                for start in args.start_layer:
                    for end in args.end_layer:   
                        # Loop to update feature mat to ignore layers between start and num_layers-end
                        # Set retain eps_threshold and scale_coff_list and get the basis
                        if mode == "baseline":
                            scale_coff_list = [0]
                            eps_threshold = None
                        elif mode == "gpm":
                            scale_coff_list = [0]
                            eps_threshold = args.gpm_eps
                        else:
                            scale_coff_list = args.scale_coff   
                            eps_threshold = None
                        for projection_location in args.projection_location:
                            # Iterate over all the projection location
                            
                            for alpha in scale_coff_list:                            
                                # Set forget eps_threshold and scale_coff_list and get the basis 
                                if mode_forget is None:
                                    mode_forget = mode
                                    scale_coff_list_forget = [alpha]
                                    eps_threshold_forget = eps_threshold
                                elif mode_forget == "baseline":
                                    scale_coff_list_forget = [0]
                                    eps_threshold_forget = None
                                elif mode_forget == "gpm":
                                    scale_coff_list_forget = [0]
                                    eps_threshold_forget = args.gpm_eps
                                else:
                                    scale_coff_list_forget =args.scale_coff_forget#[val for val in args.scale_coff_forget if alpha/1000<val]
                                    eps_threshold_forget = None
                                # Obtain the feature_matrix for retain space Mr
                                feature_retain_dict  = select_basis(full_feature_retain_dict, full_s_retain_dict, eps_threshold) 
                                feature_mat_retain_dict = get_scaled_feature_mat(feature_retain_dict, full_s_retain_dict, mode, alpha, device)  
                                terminate_alpha = False    
                                loop_best_metric = None                         
                                for alpha_forget in scale_coff_list_forget: 
                                    if terminate_alpha: 
                                        # This parameter set at the end of the loop
                                        break
                                    # Set wandb parameters 
                                    if mode==mode_forget=="baseline" and projection_type=="baseline":
                                        job_name = "baseline" 
                                    elif mode==mode_forget=="baseline" and projection_type!="baseline":
                                        continue
                                    elif projection_type=="baseline":
                                        continue
                                    elif (mode=="baseline" and mode_forget!="baseline") or (mode!="baseline" and mode_forget=="baseline"):
                                        continue
                                    else:
                                        job_name = f"{mode}({alpha})"  if not(mode=="baseline" ) else "baseline"
                                        job_name = f"{job_name}-{mode_forget}({alpha_forget})"  if not(mode_forget=="baseline") else f"{job_name}-baseline"
                                        job_name = f"{job_name}:{projection_type}" if not(projection_type =="baseline" or (mode_forget=="baseline" and mode=="baseline")) else "baseline"
                                    job_name = f"{job_name}:{start}-{num_layer-end}"
                                    # Obtain the feature_matrix for forget space Mf
                                    feature_unlearn_dict  = select_basis(full_feature_unlearn_dict, full_s_unlearn_dict, eps_threshold_forget)  
                                    feature_mat_unlearn_dict = get_scaled_feature_mat(feature_unlearn_dict, full_s_unlearn_dict, mode_forget, alpha_forget, device)                    
                                    # Get the projection matrix using Mf and Mr
                                    projection_mat = get_projections(feature_mat_retain_dict, feature_mat_unlearn_dict,projection_type, device)
                                    # Modify the projection matrix to respect the layers and the projection location in consideration (Puts identity when layer/projection location not in consideration)
                                    modified_projection_mat = {"pre":OrderedDict(), "post":OrderedDict()}
                                    for loc in projection_mat.keys():
                                        for i, act in enumerate(projection_mat[loc].keys()):
                                            # for i in range(0, num_layer):
                                            if i<start:
                                                modified_projection_mat[loc][act] = torch.eye(projection_mat[loc][act].shape[0]).to(device)
                                            elif i>num_layer-end:
                                                modified_projection_mat[loc][act] = torch.eye(projection_mat[loc][act].shape[0]).to(device)
                                            elif projection_location != "all" and (loc!=projection_location): 
                                                modified_projection_mat[loc][act] = torch.eye(projection_mat[loc][act].shape[0]).to(device)
                                            else:
                                                modified_projection_mat[loc][act] = projection_mat[loc][act]
                                            if prev_recur_proj_mat is not None:
                                                modified_projection_mat[loc][act] = torch.matmul(prev_recur_proj_mat[loc][act], modified_projection_mat[loc][act] )

                                    # Copy of the orignial trained model and project its weight.
                                    inference_model = copy.deepcopy(model)
                                    try:
                                        inference_model.module.project_weights(modified_projection_mat, args.project_classifier)
                                    except:
                                        inference_model.project_weights(modified_projection_mat, args.project_classifier)

                                    # Instantiate wandb. 
                                    run = wandb.init(
                                        # Set the project where this run will be logged
                                        project=f"Verifix-{args.dataset}-{args.project_name}",
                                        group= f"round{round}-{projection_location}layer-{args.group_name}",  
                                        name=job_name,
                                        entity=args.entity_name,
                                        dir = os.environ["LOCAL_HOME"],
                                        # Track hyperparameters and run metadata
                                        config= vars(args))
                                    # Evaluates the projection. Prints Confusion matrix and returns retain acc and forget acc.

                                    if args.use_valset:
                                        if val_loader_clean:
                                            clean_acc, clean_loss = test(inference_model, device, val_loader_clean, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Clean Val Set")
                                        full_acc, full_loss  = test(inference_model, device, val_loader, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Full Val Set")
                                    else:    
                                        if train_loader_clean:
                                            clean_acc, clean_loss = test(inference_model, device, train_loader_clean, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Clean Train Set")
                                        full_acc, full_loss  = test(inference_model, device, train_loader, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Full Train Set")
                                    test_acc, test_loss = test(inference_model, device, test_loader, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Test Set")
                                    if train_loader_corrupt:
                                        corrupt_acc, clean_loss = test(inference_model, device, train_loader_corrupt, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Corrupt Train Set")    
                                        wandb.log({ 
                                                    f"metric/corrupt_train":corrupt_acc/100,                                              
                                                    f"metric/clean_val":clean_acc/100,          
                                                    },
                                                    step=0
                                        )
                                    
                                    wandb.finish()   
                                    if train_loader_clean or val_loader_clean:      
                                        metric = clean_acc  
                                    else:
                                        metric = full_acc
                                    # if loop_best_metric is not None:                    
                                    #     if metric < loop_best_metric:
                                    #         break
                                    # else:
                                    #     loop_best_metric  =     metric             
                                    if metric > best_metric:
                                        best_metric = metric
                                        unlearnt_model = inference_model 
                                        next_recur_proj_mat =  modified_projection_mat       
    return unlearnt_model, next_recur_proj_mat
            
### Weight Norm 
def get_difference_weight_norm_classifer(model, model0, layer_names = [], unlearn_class= 0):
    distance=0
    normalization=0
    weight_norm_dict = {}
    weight_distance_dict = {}
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        # space='  ' if 'bias' in k else ''
        if k in layer_names:
            if k == layer_names[-1]:
                weight_diff =  (p.data-p0.data).pow(2)
                weight_diff[unlearn_class] = torch.zeros_like(weight_diff[unlearn_class]) 
            else:
                weight_diff = (p.data-p0.data).pow(2)
            current_dist= weight_diff.sum().item()
            current_norm=p.data.pow(2).sum().item()
            distance+=current_dist
            normalization+=current_norm
            weight_distance_dict[k] = 1.0*np.sqrt(current_dist) 
            weight_norm_dict[k] = 1.0*np.sqrt(current_dist/current_norm)
    print(f'Distance: {np.sqrt(distance)}')
    print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    weight_norm_dict["total" ] = 1.0*np.sqrt(distance/normalization)
    weight_distance_dict["total" ] = 1.0*np.sqrt(distance)
    return weight_distance_dict, weight_norm_dict 

def get_difference_weight_norm(model, model0, layer_names = []):
    distance=0
    normalization=0
    weight_norm_dict = {}
    weight_distance_dict = {}
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        if k in layer_names:
            weight_diff = (p.data-p0.data).absolute()
            current_dist= weight_diff.sum().item()
            current_norm= p.numel()
            distance+=current_dist
            normalization+=current_norm
            weight_distance_dict[k] = 1.0*np.sqrt(current_dist) 
            weight_norm_dict[k] = 1.0*np.sqrt(current_dist/current_norm)
    print(f'Distance: {np.sqrt(distance)}')
    print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    weight_norm_dict["total" ] = 1.0*np.sqrt(distance/normalization)
    weight_distance_dict["total" ] = 1.0*np.sqrt(distance)
    return weight_distance_dict, weight_norm_dict 