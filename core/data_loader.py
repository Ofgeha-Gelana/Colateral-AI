# core/data_loader.py
import pandas as pd

# --- Master Data Functions ---
def get_branches_data():
    # ... (data collapsed for brevity)
    return {
        'Adama ': ['Adama', 'Abomsa', 'Arboye', 'Adelle', 'Asala', 'Awash Malkasa', 'Awash Sebat Kilo', 'Bale Gasgar', 'Batu', 'Boqoji', 'Boku Shanan', 'Bole Nura Hera', 'Bosat', 'Bote', 'Chilalo', 'Chole', 'Darartu Tullu', 'Diksis', 'Dhaka Adi', 'Dodota', 'Ela Bal\'a', 'Etaya', 'Gololcha', 'Hawas', 'Huruta', 'Habe', 'Maqi', 'Mojo', 'Nagelle Arbagugu', 'Sagure', 'shirka', 'Sire Arsi', 'Siltana', 'Torban Obo', 'Robe Didea', 'Karayu', 'Kula', 'Qersa Arsi', 'Qurqura', 'Xicho', 'Wanji', 'Dabe soloke', 'Boru Jawi', 'Dadacha Arara', 'Aba Jama', 'Meraro', 'Qechema', 'Chaffe', 'Bulbula'],
        'Central Merkato ': ['Aba Mala', 'Aba Nabso', 'Addisu Gabaya', 'Agamsa', 'Bakanisa', 'Bomb Tera', 'Birbirsa', 'Chalchali', 'Churchill', 'Dollo Bidena', 'Finfinne', 'Gofa Gabriel', 'Gofa Mazoria', 'Gojam Berenda', 'Gulale', 'Meskel Flower', 'Mesalemia', 'Merkato', 'Oda', 'Qarsa', 'Shagar', 'Sulula Garbi', 'Tullu Ejersa', 'Lafto', 'Kolfe', 'Wayu', 'Wirtu Merkato', "Ya'I Gulale", 'Tufa Muna', 'Caffee Arara', 'Marde', 'Bilisumm', 'Rufe Rarre', 'Mujja', 'Adami', 'Shanan Jida', 'karra Xuxo'],
        'Dire Dawa': ['Afranqalo', 'Aw-Abadir', 'Babile', 'Badano', 'Bedessa', 'Boke Tiko', 'Chalanqo', 'Chiro', 'Chinaqsan', 'Deder', 'Dire Dawa', 'Gelemso', 'Gode', 'Hafatesa', 'Haromaya', 'Hirna', 'Hudane', 'Jigjiga', 'Kali', 'kereyu', 'Kombolcha Hararge', 'Laga Oda', 'Melka Balo', 'Sabian', 'Togochale'],
        'Eastern Finfine': ["Abiebe Tufa", "Ada'a", "Alaltu", "Aqaqi", "Bake", "Bole Lami", "Bole Bulbula", "Bole Medhanialem", "Bole Atlas", "Bole Mikael", "Bole Rwanda", "CMC", "Dale Dembel", "Erer", "Eka Abbadho", "Ferensay legasiyon", "Gelan", "Gerji", "Gotera", "Gurd shola", "Gumbichu", "Haile Gebre", "Hayahulet Mazoria", "Hora Arsadi", "Karra Qallu", "Kilole", "Laga Xafo", "Lamberet", "Lume", "Oda Nabe", "Tirunesh Dibaba", "Saris", "Sandafa Bake", "Shano", "Saden Eka", "Kotobe", "Qabana", "Bilile", "Handhura Tulama", "Karra Allo", "Bole Arabsa", "Gawasa", "Abichu Nya'a"],
        'Jimma': ['Abba Jifar', 'Agaro', 'Alge Sechi', 'Asendabo', 'Bedele', 'Chora', 'Darimu', 'Dembi', 'Gambela', 'Gera', 'Gore', 'Jiren', 'Limu Genet', 'Limmu Saqqa', 'Mizan Aman', 'Saglan Ilu', 'Shanan Gibe', 'Tepi', 'Yayo', 'Sokoru', 'Gachi', 'Shabe', 'Awetu', 'Omo Nada', 'Hirmata', 'Mattu', 'Oda Karra', 'Chanqa Birbir', 'Hawa Galan', 'Tullu Gana', 'Bake Jama'],
        'Nekemte': ['Angar Gute', 'Asosa', 'Bako Tibe', 'Dambi Dollo', 'Naqamte', 'Gida Ayana', 'Gimbi', 'Guduru', 'Gulliso', 'Najjo', 'Shambu', 'Leqa', 'Hababo Guduru', 'Nunu Qumba', 'Mandi', 'Begi', 'Ganji', 'Ukke', 'Muka Arara', 'Jimma Arjo', 'Sibu Sire', 'Qeto Machara', 'Anfilo', 'Qilxu-kara', 'Jimma Rare', 'Ayira', 'Qebe', 'Agamsa Migir', 'Bonya Boshe', 'Gudatu Arjo'],
        'Northern ': ['Bahir Dar', 'Dessie', 'Djibruh', 'Shire', 'Humara', 'Metemma', 'Mekele', 'Kamise', 'Gonder', 'Genda Wuha'],
        'Shashemene': ['Adaba', 'Arsi Nagele', 'Agarfa', 'Ajje', 'Awasho', 'Dalo Mana', 'Dodola', 'Goba', 'Gasara', 'Ginnir', 'Goro', 'Harufa', 'Harana Buluq', 'Haro Dumal', 'Hasasa', 'Kofale', 'Mada walabu', 'Robe', 'Shashamane', 'Uta Wayyu', 'Warqa', 'General Waqo Gutu', 'Jara', 'Ali', 'Dalo Sabro', 'Oda Bilibila', 'Burqa Walabu', 'Bishan Guracha', 'Mida Qagni'],
        'Western Finfine': ['Ayer Tena', 'Ambo', 'Asko', "Ada'a Barga", 'Bethel', 'Burayu', 'Chancho', 'Dara', 'Dawo', 'Dire Enchini', 'Ejere', 'Fiche', 'Furi', 'Ginchi', 'Gedo', 'Gafarsa', 'Gudar', 'Guje', 'Ilu Galan', 'Jahan Amaya', 'Jamo', 'Kuyu', 'Leman', 'Sabbata Hawas', 'Sabbata Ayo', 'Sululta', 'Shararo', 'Sodo Dachi', 'Holata', 'Tullu Bollo', 'Meta Walqixe', 'Muka Turi', 'Mulo', 'Mogor', 'Nono', 'Wolkite', 'Waliso', 'Wasarbi', 'Liban Jawi', 'Oda Bisil', 'Wara Jarso', 'Ayetu', 'Ochocha', 'Odo Liban', 'Shufune', 'Olankomi', 'Asgori', 'Malka Gafarsa', 'Qorke Karabu', 'Torban Wale', 'Saglan Eggu', 'Karra Qore', 'Gafarsa Nono', 'Kutaye', 'Goro Qeransa', 'Abuna', 'Jaldu', 'Kombe', 'Saden Sodo', 'Labu', 'Dire Sololiya', 'Tajji', 'Yaya Gullele', 'Boku Migira Bake', 'Qajela Doyo', 'Bantu']
    }

def get_building_rates_data():
    # ... (data collapsed for brevity)
    return [
        {'Building Type': 'G+1 and G+2', 'Excellent_Min': 28513, 'Excellent_Max': 31517, 'Good_Min': 23969, 'Good_Max': 26503, 'Average_Min': 18679, 'Average_Max': 20663, 'Economy_Min': 13967, 'Economy_Max': 15428, 'Minimum_Min': 11847, 'Minimum_Max': 13087},
        {'Building Type': 'G+3 and G+4', 'Excellent_Min': 31464, 'Excellent_Max': 34797, 'Good_Min': 25622, 'Good_Max': 28321, 'Average_Min': 21352, 'Average_Max': 23611, 'Economy_Min': 16199, 'Economy_Max': 17908, 'Minimum_Min': 14822, 'Minimum_Max': 16393},
        {'Building Type': 'G+5 and G+6', 'Excellent_Min': 32205, 'Excellent_Max': 35595, 'Good_Min': 25840, 'Good_Max': 28568, 'Average_Min': 21158, 'Average_Max': 23389, 'Economy_Min': 15869, 'Economy_Max': 17522, 'Minimum_Min': 13995, 'Minimum_Max': 15042},
        {'Building Type': 'G+7 and Above', 'Excellent_Min': 31242, 'Excellent_Max': 34519, 'Good_Min': 27634, 'Good_Max': 30525, 'Average_Min': 24989, 'Average_Max': 27605, 'Economy_Min': 21985, 'Economy_Max': 24299, 'Minimum_Min': 20938, 'Minimum_Max': 22260},
        {'Building Type': 'Single Story Building (higher Villa)', 'Excellent_Min': 32900, 'Excellent_Max': 36358, 'Good_Min': 23490, 'Good_Max': 25960, 'Average_Min': 16351, 'Average_Max': 18080, 'Economy_Min': 11486, 'Economy_Max': 12696, 'Minimum_Min': 5582, 'Minimum_Max': 6175},
    ]

def get_component_percentages():
    # ... (data collapsed for brevity)
    data = {
        'Building_Component': ['Sub Structure', 'Super Structure', 'Block work- Internal', 'Block work- External', 'Roofing', 'Joinery', 'Metal Work', 'Floor finish', 'Wall finish - Internal', 'Wall finish - External', 'Ceiling works', 'Hand rail and Balustrade', 'Sanitary Work', 'Electrical Work'],
        'Single_Storey_Poor': [0.13, 0.00, 0.04, 0.03, 0.17, 0.03, 0.20, 0.03, 0.14, 0.07, 0.04, 0.00, 0.08, 0.04],
        'Single_Storey_Avg': [0.12, 0.06, 0.05, 0.04, 0.10, 0.06, 0.16, 0.10, 0.08, 0.08, 0.06, 0.00, 0.06, 0.03],
        'Single_Storey_Best': [0.07, 0.06, 0.04, 0.06, 0.03, 0.10, 0.09, 0.10, 0.10, 0.18, 0.11, 0.00, 0.04, 0.02],
        'G1_G2_Poor': [0.29, 0.33, 0.03, 0.07, 0.04, 0.01, 0.09, 0.02, 0.02, 0.01, 0.00, 0.01, 0.04, 0.04],
        'G1_G2_Avg': [0.19, 0.22, 0.03, 0.09, 0.05, 0.02, 0.07, 0.06, 0.10, 0.06, 0.04, 0.01, 0.03, 0.03],
        'G1_G2_Best': [0.08, 0.09, 0.05, 0.13, 0.08, 0.03, 0.05, 0.06, 0.18, 0.10, 0.08, 0.03, 0.02, 0.02],
        'G3_G4_Poor': [0.27, 0.41, 0.03, 0.04, 0.02, 0.01, 0.09, 0.02, 0.02, 0.00, 0.00, 0.01, 0.03, 0.05],
        'G3_G4_Avg': [0.19, 0.29, 0.02, 0.05, 0.03, 0.02, 0.07, 0.07, 0.10, 0.04, 0.04, 0.01, 0.03, 0.04],
        'G3_G4_Best': [0.10, 0.16, 0.03, 0.05, 0.08, 0.02, 0.06, 0.09, 0.21, 0.07, 0.08, 0.00, 0.02, 0.03]
    }
    df = pd.DataFrame(data)
    df.set_index('Building_Component', inplace=True)
    return df

def get_all_location_data():
    # ... (data collapsed for brevity)
    return {
        'Finfinne Border A1': {
            'Residential': {
                '1st': { (0, 200): 24000, (201, 400): 19200, (401, 600): 15360, (601, 800): 11520, (801, 1000): 8640, (1001, 1500): 6480, (1501, 2000): 4860, (2001, 3000): 3640, (3001, 5000): 2720, (5001, 10000): 1400, (10001, float('inf')): 700 },
                '2nd': { (0, 200): 18000, (201, 400): 14400, (401, 600): 11520, (601, 800): 8640, (801, 1000): 6480, (1001, 1500): 4860, (1501, 2000): 3640, (2001, 3000): 2730, (3001, 5000): 2040, (5001, 10000): 1000, (10001, float('inf')): 500 },
                '3rd': { (0, 200): 13500, (201, 400): 10800, (401, 600): 8640, (601, 800): 6480, (801, 1000): 4860, (1001, 1500): 3640, (1501, 2000): 2730, (2001, 3000): 2050, (3001, 5000): 1540, (5001, 10000): 680, (10001, float('inf')): 340 },
                '4th': { (0, 200): 10120, (201, 400): 8100, (401, 600): 6480, (601, 800): 4860, (801, 1000): 3640, (1001, 1500): 2720, (1501, 2000): 2040, (2001, 3000): 1540, (3001, 5000): 1140, (5001, 10000): 480, (10001, float('inf')): 240 },
            },
            'Commercial': {
                '1st': { (0, 200): 28000, (201, 400): 22400, (401, 600): 17920, (601, 800): 13440, (801, 1000): 10080, (1001, 1500): 7560, (1501, 2000): 5670, (2001, 3000): 4250, (3001, 5000): 3180, (5001, 10000): 1600, (10001, float('inf')): 800 },
                '2nd': { (0, 200): 21000, (201, 400): 16800, (401, 600): 13440, (601, 800): 10080, (801, 1000): 7560, (1001, 1500): 5670, (1501, 2000): 4250, (2001, 3000): 3180, (3001, 5000): 2380, (5001, 10000): 1200, (10001, float('inf')): 600 },
                '3rd': { (0, 200): 15750, (201, 400): 12600, (401, 600): 10080, (601, 800): 7560, (801, 1000): 5680, (1001, 1500): 4240, (1501, 2000): 3190, (2001, 3000): 2390, (3001, 5000): 1800, (5001, 10000): 900, (10001, float('inf')): 450 },
                '4th': { (0, 200): 11820, (201, 400): 11460, (401, 600): 7560, (601, 800): 5670, (801, 1000): 4260, (1001, 1500): 3180, (1501, 2000): 2380, (2001, 3000): 1790, (3001, 5000): 1340, (5001, 10000): 680, (10001, float('inf')): 340 },
            }
        },
        'Surrounding Finfine B1': {
            'Residential': {
                '1st': { (0, 200): 12000, (201, 400): 9600, (401, 600): 7680, (601, 800): 5760, (801, 1000): 4320, (1001, 1500): 3240, (1501, 2000): 2430, (2001, 3000): 1820, (3001, 5000): 1360, (5001, 10000): 680, (10001, float('inf')): 340 },
                '2nd': { (0, 200): 9000, (201, 400): 7200, (401, 600): 5760, (601, 800): 4320, (801, 1000): 3240, (1001, 1500): 2430, (1501, 2000): 1820, (2001, 3000): 1370, (3001, 5000): 1020, (5001, 10000): 500, (10001, float('inf')): 240 },
                '3rd': { (0, 200): 6750, (201, 400): 5400, (401, 600): 4320, (601, 800): 3240, (801, 1000): 2430, (1001, 1500): 1820, (1501, 2000): 1360, (2001, 3000): 1020, (3001, 5000): 760, (5001, 10000): 380, (10001, float('inf')): 180 },
                '4th': { (0, 200): 5060, (201, 400): 4048, (401, 600): 3240, (601, 800): 2430, (801, 1000): 1820, (1001, 1500): 1360, (1501, 2000): 1020, (2001, 3000): 760, (3001, 5000): 580, (5001, 10000): 280, (10001, float('inf')): 140 },
            },
            'Commercial': {
                '1st': { (0, 200): 13000, (201, 400): 10400, (401, 600): 8320, (601, 800): 6240, (801, 1000): 4680, (1001, 1500): 3510, (1501, 2000): 2630, (2001, 3000): 1980, (3001, 5000): 1480, (5001, 10000): 740, (10001, float('inf')): 360 },
                '2nd': { (0, 200): 9750, (201, 400): 7800, (401, 600): 6240, (601, 800): 4680, (801, 1000): 3510, (1001, 1500): 2630, (1501, 2000): 1980, (2001, 3000): 1480, (3001, 5000): 1100, (5001, 10000): 540, (10001, float('inf')): 260 },
                '3rd': { (0, 200): 7320, (201, 400): 5860, (401, 600): 4680, (601, 800): 3510, (801, 1000): 2630, (1001, 1500): 1980, (1501, 2000): 1480, (2001, 3000): 1100, (3001, 5000): 820, (5001, 10000): 400, (10001, float('inf')): 200 },
                '4th': { (0, 200): 5480, (201, 400): 4380, (401, 600): 3510, (601, 800): 2630, (801, 1000): 1980, (1001, 1500): 1480, (1501, 2000): 1100, (2001, 3000): 820, (3001, 5000): 600, (5001, 10000): 300, (10001, float('inf')): 150 },
            }
        },
        'Surrounding Finfine B2': {
            'Residential': {
                '1st': { (0, 200): 7650, (201, 400): 6120, (401, 600): 4896, (601, 800): 3230, (801, 1000): 2423, (1001, 1500): 1811, (1501, 2000): 1360, (2001, 3000): 1020, (3001, 5000): 765, (5001, 10000): 374, (10001, float('inf')): 187 },
                '2nd': { (0, 200): 5126, (201, 400): 4100, (401, 600): 3281, (601, 800): 2159, (801, 1000): 1624, (1001, 1500): 1216, (1501, 2000): 910, (2001, 3000): 680, (3001, 5000): 510, (5001, 10000): 255, (10001, float('inf')): 128 },
                '3rd': { (0, 200): 3281, (201, 400): 2623, (401, 600): 2100, (601, 800): 1386, (801, 1000): 1037, (1001, 1500): 774, (1501, 2000): 587, (2001, 3000): 434, (3001, 5000): 323, (5001, 10000): 153, (10001, float('inf')): 77 },
                '4th': { (0, 200): 2100, (201, 400): 1666, (401, 600): 1343, (601, 800): 884, (801, 1000): 663, (1001, 1500): 502, (1501, 2000): 374, (2001, 3000): 281, (3001, 5000): 204, (5001, 10000): 102, (10001, float('inf')): 51 },
            },
            'Commercial': {
                '1st': { (0, 200): 8500, (201, 400): 6800, (401, 600): 5440, (601, 800): 3587, (801, 1000): 2686, (1001, 1500): 2015, (1501, 2000): 1513, (2001, 3000): 1131, (3001, 5000): 850, (5001, 10000): 383, (10001, float('inf')): 187 },
                '2nd': { (0, 200): 5695, (201, 400): 4556, (401, 600): 3647, (601, 800): 2406, (801, 1000): 1802, (1001, 1500): 1352, (1501, 2000): 1012, (2001, 3000): 765, (3001, 5000): 578, (5001, 10000): 289, (10001, float('inf')): 136 },
                '3rd': { (0, 200): 3647, (201, 400): 2917, (401, 600): 2329, (601, 800): 1539, (801, 1000): 1156, (1001, 1500): 867, (1501, 2000): 651, (2001, 3000): 485, (3001, 5000): 357, (5001, 10000): 170, (10001, float('inf')): 85 },
                '4th': { (0, 200): 2329, (201, 400): 1867, (401, 600): 1496, (601, 800): 986, (801, 1000): 740, (1001, 1500): 553, (1501, 2000): 417, (2001, 3000): 315, (3001, 5000): 221, (5001, 10000): 111, (10001, float('inf')): 51 },
            }
        },
        'Surrounding Finfine B3': {
            'Residential': {
                '1st': { (0, 200): 6000, (201, 400): 4800, (401, 600): 3840, (601, 800): 2535, (801, 1000): 1898, (1001, 1500): 1425, (1501, 2000): 1065, (2001, 3000): 803, (3001, 5000): 600, (5001, 10000): 300, (10001, float('inf')): 150 },
                '2nd': { (0, 200): 4020, (201, 400): 3216, (401, 600): 2573, (601, 800): 1695, (801, 1000): 1275, (1001, 1500): 953, (1501, 2000): 713, (2001, 3000): 533, (3001, 5000): 405, (5001, 10000): 195, (10001, float('inf')): 98 },
                '3rd': { (0, 200): 2573, (201, 400): 2055, (401, 600): 1646, (601, 800): 1088, (801, 1000): 810, (1001, 1500): 608, (1501, 2000): 458, (2001, 3000): 345, (3001, 5000): 255, (5001, 10000): 120, (10001, float('inf')): 60 },
                '4th': { (0, 200): 1635, (201, 400): 1305, (401, 600): 1047, (601, 800): 698, (801, 1000): 518, (1001, 1500): 390, (1501, 2000): 293, (2001, 3000): 218, (3001, 5000): 150, (5001, 10000): 75, (10001, float('inf')): 38 },
            },
            'Commercial': {
                '1st': { (0, 200): 6750, (201, 400): 5400, (401, 600): 4320, (601, 800): 2850, (801, 1000): 2138, (1001, 1500): 1598, (1501, 2000): 1200, (2001, 3000): 900, (3001, 5000): 675, (5001, 10000): 338, (10001, float('inf')): 165 },
                '2nd': { (0, 200): 4523, (201, 400): 3618, (401, 600): 2895, (601, 800): 1905, (801, 1000): 1433, (1001, 1500): 1073, (1501, 2000): 803, (2001, 3000): 600, (3001, 5000): 450, (5001, 10000): 225, (10001, float('inf')): 113 },
                '3rd': { (0, 200): 2895, (201, 400): 2318, (401, 600): 1853, (601, 800): 1223, (801, 1000): 915, (1001, 1500): 683, (1501, 2000): 518, (2001, 3000): 383, (3001, 5000): 285, (5001, 10000): 135, (10001, float('inf')): 88 },
                '4th': { (0, 200): 959, (201, 400): 770, (401, 600): 616, (601, 800): 462, (801, 1000): 343, (1001, 1500): 259, (1501, 2000): 196, (2001, 3000): 147, (3001, 5000): 112, (5001, 10000): 56, (10001, float('inf')): 28 },
            }
        },
        'Major Cities C1': {
            'Residential': {
                '1st': { (0, 200): 11500, (201, 400): 9200, (401, 600): 7360, (601, 800): 5520, (801, 1000): 4140, (1001, 1500): 3105, (1501, 2000): 2323, (2001, 3000): 1748, (3001, 5000): 1311, (5001, 10000): 644, (10001, float('inf')): 322 },
                '2nd': { (0, 200): 7705, (201, 400): 6164, (401, 600): 4934, (601, 800): 3703, (801, 1000): 2772, (1001, 1500): 2082, (1501, 2000): 1564, (2001, 3000): 1173, (3001, 5000): 874, (5001, 10000): 437, (10001, float('inf')): 207 },
                '3rd': { (0, 200): 4934, (201, 400): 3945, (401, 600): 3151, (601, 800): 2369, (801, 1000): 1771, (1001, 1500): 1334, (1501, 2000): 1001, (2001, 3000): 748, (3001, 5000): 552, (5001, 10000): 276, (10001, float('inf')): 138 },
                '4th': { (0, 200): 3151, (201, 400): 2530, (401, 600): 2024, (601, 800): 1518, (801, 1000): 1139, (1001, 1500): 851, (1501, 2000): 644, (2001, 3000): 483, (3001, 5000): 345, (5001, 10000): 173, (10001, float('inf')): 81 },
            },
            'Commercial': {
                '1st': { (0, 200): 13800, (201, 400): 11040, (401, 600): 8832, (601, 800): 6624, (801, 1000): 4968, (1001, 1500): 3726, (1501, 2000): 4865, (2001, 3000): 2093, (3001, 5000): 1564, (5001, 10000): 782, (10001, float('inf')): 391 },
                '2nd': { (0, 200): 9246, (201, 400): 7397, (401, 600): 5916, (601, 800): 4439, (801, 1000): 3335, (1001, 1500): 2496, (1501, 2000): 1875, (2001, 3000): 1403, (3001, 5000): 1035, (5001, 10000): 506, (10001, float('inf')): 253 },
                '3rd': { (0, 200): 5911, (201, 400): 4727, (401, 600): 3784, (601, 800): 2841, (801, 1000): 2128, (1001, 1500): 1599, (1501, 2000): 1196, (2001, 3000): 897, (3001, 5000): 667, (5001, 10000): 334, (10001, float('inf')): 173 },
                '4th': { (0, 200): 3784, (201, 400): 3025, (401, 600): 2415, (601, 800): 1817, (801, 1000): 1357, (1001, 1500): 1024, (1501, 2000): 771, (2001, 3000): 575, (3001, 5000): 414, (5001, 10000): 207, (10001, float('inf')): 104 },
            }
        },
        'Major Cities C2': {
            'Residential': {
                '1st': { (0, 200): 6300, (201, 400): 5040, (401, 600): 4032, (601, 800): 3024, (801, 1000): 2268, (1001, 1500): 1701, (1501, 2000): 1274, (2001, 3000): 952, (3001, 5000): 700, (5001, 10000): 350, (10001, float('inf')): 175 },
                '2nd': { (0, 200): 4221, (201, 400): 3377, (401, 600): 2702, (601, 800): 2016, (801, 1000): 1519, (1001, 1500): 1120, (1501, 2000): 840, (2001, 3000): 630, (3001, 5000): 462, (5001, 10000): 224, (10001, float('inf')): 154 },
                '3rd': { (0, 200): 2702, (201, 400): 2163, (401, 600): 1722, (601, 800): 1288, (801, 1000): 973, (1001, 1500): 728, (1501, 2000): 546, (2001, 3000): 406, (3001, 5000): 308, (5001, 10000): 154, (10001, float('inf')): 126 },
                '4th': { (0, 200): 1729, (201, 400): 1386, (401, 600): 1106, (601, 800): 826, (801, 1000): 630, (1001, 1500): 462, (1501, 2000): 350, (2001, 3000): 259, (3001, 5000): 196, (5001, 10000): 98, (10001, float('inf')): 56 },
            },
            'Commercial': {
                '1st': { (0, 200): 7000, (201, 400): 5600, (401, 600): 4480, (601, 800): 3360, (801, 1000): 2520, (1001, 1500): 1890, (1501, 2000): 1417, (2001, 3000): 1064, (3001, 5000): 798, (5001, 10000): 364, (10001, float('inf')): 182 },
                '2nd': { (0, 200): 4690, (201, 400): 3752, (401, 600): 3003, (601, 800): 2251, (801, 1000): 1688, (1001, 1500): 1267, (1501, 2000): 952, (2001, 3000): 714, (3001, 5000): 532, (5001, 10000): 252, (10001, float('inf')): 161 },
                '3rd': { (0, 200): 2996, (201, 400): 2394, (401, 600): 1918, (601, 800): 1442, (801, 1000): 1078, (1001, 1500): 812, (1501, 2000): 602, (2001, 3000): 455, (3001, 5000): 336, (5001, 10000): 168, (10001, float('inf')): 140 },
                '4th': { (0, 200): 1918, (201, 400): 1526, (401, 600): 1232, (601, 800): 924, (801, 1000): 686, (1001, 1500): 518, (1501, 2000): 392, (2001, 3000): 294, (3001, 5000): 210, (5001, 10000): 105, (10001, float('inf')): 42 },
            }
        },
        'Secondary Major Cities D1': {
            'Residential': {
                '1st': { (0, 200): 3500, (201, 400): 2800, (401, 600): 2240, (601, 800): 1680, (801, 1000): 1260, (1001, 1500): 945, (1501, 2000): 700, (2001, 3000): 532, (3001, 5000): 392, (5001, 10000): 196, (10001, float('inf')): 98 },
                '2nd': { (0, 200): 2345, (201, 400): 1876, (401, 600): 1498, (601, 800): 1127, (801, 1000): 847, (1001, 1500): 630, (1501, 2000): 476, (2001, 3000): 357, (3001, 5000): 266, (5001, 10000): 126, (10001, float('inf')): 63 },
                '3rd': { (0, 200): 1501, (201, 400): 1204, (401, 600): 959, (601, 800): 721, (801, 1000): 539, (1001, 1500): 406, (1501, 2000): 301, (2001, 3000): 224, (3001, 5000): 168, (5001, 10000): 84, (10001, float('inf')): 42 },
                '4th': { (0, 200): 966, (201, 400): 770, (401, 600): 616, (601, 800): 462, (801, 1000): 343, (1001, 1500): 259, (1501, 2000): 196, (2001, 3000): 147, (3001, 5000): 112, (5001, 10000): 56, (10001, float('inf')): 28 },
            },
            'Commercial': {
                '1st': { (0, 200): 4200, (201, 400): 3360, (401, 600): 2688, (601, 800): 2016, (801, 1000): 1512, (1001, 1500): 1134, (1501, 2000): 847, (2001, 3000): 637, (3001, 5000): 476, (5001, 10000): 238, (10001, float('inf')): 112 },
                '2nd': { (0, 200): 2814, (201, 400): 2254, (401, 600): 1799, (601, 800): 1351, (801, 1000): 1015, (1001, 1500): 756, (1501, 2000): 567, (2001, 3000): 427, (3001, 5000): 322, (5001, 10000): 154, (10001, float('inf')): 77 },
                '3rd': { (0, 200): 1799, (201, 400): 1442, (401, 600): 1155, (601, 800): 861, (801, 1000): 651, (1001, 1500): 483, (1501, 2000): 384, (2001, 3000): 273, (3001, 5000): 196, (5001, 10000): 98, (10001, float('inf')): 49 },
                '4th': { (0, 200): 1148, (201, 400): 917, (401, 600): 735, (601, 800): 553, (801, 1000): 413, (1001, 1500): 308, (1501, 2000): 231, (2001, 3000): 175, (3001, 5000): 126, (5001, 10000): 63, (10001, float('inf')): 28 },
            }
        },
        'Secondary Major Cities D2': {
            'Residential': {
                '1st': { (0, 200): 2800, (201, 400): 2240, (401, 600): 1792, (601, 800): 1344, (801, 1000): 1008, (1001, 1500): 728, (1501, 2000): 567, (2001, 3000): 847, (3001, 5000): 308, (5001, 10000): 154, (10001, float('inf')): 77 },
                '2nd': { (0, 200): 1876, (201, 400): 1498, (401, 600): 1197, (601, 800): 896, (801, 1000): 672, (1001, 1500): 504, (1501, 2000): 378, (2001, 3000): 287, (3001, 5000): 210, (5001, 10000): 105, (10001, float('inf')): 49 },
                '3rd': { (0, 200): 1204, (201, 400): 959, (401, 600): 770, (601, 800): 574, (801, 1000): 434, (1001, 1500): 322, (1501, 2000): 245, (2001, 3000): 182, (3001, 5000): 126, (5001, 10000): 63, (10001, float('inf')): 31 },
                '4th': { (0, 200): 770, (201, 400): 616, (401, 600): 490, (601, 800): 371, (801, 1000): 280, (1001, 1500): 210, (1501, 2000): 154, (2001, 3000): 119, (3001, 5000): 84, (5001, 10000): 42, (10001, float('inf')): 21 },
            },
            'Commercial': {
                '1st': { (0, 200): 3500, (201, 400): 2600, (401, 600): 2240, (601, 800): 1680, (801, 1000): 1260, (1001, 1500): 945, (1501, 2000): 707, (2001, 3000): 532, (3001, 5000): 399, (5001, 10000): 196, (10001, float('inf')): 98 },
                '2nd': { (0, 200): 2345, (201, 400): 1876, (401, 600): 1498, (601, 800): 1127, (801, 1000): 847, (1001, 1500): 630, (1501, 2000): 476, (2001, 3000): 357, (3001, 5000): 266, (5001, 10000): 126, (10001, float('inf')): 63 },
                '3rd': { (0, 200): 1501, (201, 400): 1197, (401, 600): 959, (601, 800): 721, (801, 1000): 539, (1001, 1500): 406, (1501, 2000): 301, (2001, 3000): 224, (3001, 5000): 168, (5001, 10000): 84, (10001, float('inf')): 42 },
                '4th': { (0, 200): 959, (201, 400): 770, (401, 600): 616, (601, 800): 462, (801, 1000): 343, (1001, 1500): 259, (1501, 2000): 196, (2001, 3000): 147, (3001, 5000): 112, (5001, 10000): 56, (10001, float('inf')): 28 },
            }
        },
        'Tertiary Towns E1': {
            'Residential': {
                '1st': { (0, 200): 980, (201, 400): 735, (401, 600): 546, (601, 800): 406, (801, 1000): 308, (1001, 1500): 231, (1501, 2000): 175, (2001, 3000): 126, (3001, 5000): 98, (5001, 10000): 49, (10001, float('inf')): 24 },
                '2nd': { (0, 200): 658, (201, 400): 490, (401, 600): 371, (601, 800): 280, (801, 1000): 210, (1001, 1500): 154, (1501, 2000): 119, (2001, 3000): 84, (3001, 5000): 63, (5001, 10000): 31, (10001, float('inf')): 15 },
                '3rd': { (0, 200): 420, (201, 400): 315, (401, 600): 238, (601, 800): 175, (801, 1000): 133, (1001, 1500): 98, (1501, 2000): 77, (2001, 3000): 56, (3001, 5000): 42, (5001, 10000): 21, (10001, float('inf')): 10 },
                '4th': { (0, 200): 266, (201, 400): 196, (401, 600): 147, (601, 800): 112, (801, 1000): 84, (1001, 1500): 63, (1501, 2000): 49, (2001, 3000): 35, (3001, 5000): 28, (5001, 10000): 14, (10001, float('inf')): 7 },
            },
            'Commercial': {
                '1st': { (0, 200): 1260, (201, 400): 945, (401, 600): 707, (601, 800): 518, (801, 1000): 399, (1001, 1500): 301, (1501, 2000): 224, (2001, 3000): 168, (3001, 5000): 126, (5001, 10000): 63, (10001, float('inf')): 31 },
                '2nd': { (0, 200): 840, (201, 400): 630, (401, 600): 469, (601, 800): 357, (801, 1000): 266, (1001, 1500): 196, (1501, 2000): 147, (2001, 3000): 112, (3001, 5000): 84, (5001, 10000): 42, (10001, float('inf')): 21 },
                '3rd': { (0, 200): 539, (201, 400): 406, (401, 600): 391, (601, 800): 224, (801, 1000): 168, (1001, 1500): 126, (1501, 2000): 91, (2001, 3000): 70, (3001, 5000): 49, (5001, 10000): 24, (10001, float('inf')): 11 },
                '4th': { (0, 200): 350, (201, 400): 259, (401, 600): 196, (601, 800): 147, (801, 1000): 112, (1001, 1500): 84, (1501, 2000): 63, (2001, 3000): 49, (3001, 5000): 35, (5001, 10000): 17, (10001, float('inf')): 8 },
            }
        },
        'Tertiary Towns E2': {
            'Residential': {
                '1st': { (0, 200): 560, (201, 400): 420, (401, 600): 315, (601, 800): 238, (801, 1000): 175, (1001, 1500): 133, (1501, 2000): 98, (2001, 3000): 77, (3001, 5000): 56, (5001, 10000): 28, (10001, float('inf')): 14 },
                '2nd': { (0, 200): 378, (201, 400): 280, (401, 600): 210, (601, 800): 161, (801, 1000): 119, (1001, 1500): 91, (1501, 2000): 70, (2001, 3000): 49, (3001, 5000): 35, (5001, 10000): 17, (10001, float('inf')): 8 },
                '3rd': { (0, 200): 238, (201, 400): 175, (401, 600): 133, (601, 800): 98, (801, 1000): 77, (1001, 1500): 56, (1501, 2000): 42, (2001, 3000): 28, (3001, 5000): 24, (5001, 10000): 11, (10001, float('inf')): 6 },
                '4th': { (0, 200): 154, (201, 400): 112, (401, 600): 84, (601, 800): 63, (801, 1000): 49, (1001, 1500): 35, (1501, 2000): 28, (2001, 3000): 21, (3001, 5000): 14, (5001, 10000): 7, (10001, float('inf')): 4 },
            }
        }
    }

def get_materials_by_category(category: str):
    # ... (function is unchanged)
    villa_and_multi_story = {
        "Foundation": ["RC, Best workmanship", "RC, Good workmanship", "RC, Stone Masonry", "Stone Masonry"],
        "Roofing": ["Decra, RC, EGA & similar tiles", "CIS G 28, CIS G 32, GQ", "CIS G32", "CIS G 32 & 35"],
        "Metal Work": ["Aluminum profile", "Aluminum imitation", "LTZ"],
        "Floor": ["Granite, Marble, ceramic, parquet, porcelain", "Parquet, Ceramic, All GQ", "PVC, Terrazzo, Cement tiles", "Cement tiles, Sc."],
        "Ceiling": ["Gypsum, PVC, Parquet", "Good quality gypsum, PVC", "Chip wood, Hard Board, Ply wood", "Abujidi"],
        "Sanitary": ["Jacuzzi, Steam, Sauna", "Good quality HWB, WC etc.", "Shower, HWB, WC (all MQ)", "Dry latrine, Turkish seat"]
    }
    mph_factory = {
        "Foundation": ["RC, Best workmanship", "RC, Good workmanship", "RC, Stone Masonry", "Stone Masonry"],
        "Structure": ["RC, Steel Structure, Best workmanship", "RC, Steel Structure, Good workmanship", "RC, (Average)", "RC, Wooden structure"],
        "Roofing": ["EGA on steel, RC Slab", "CIS G 28, CIS G 32, on steel GQ", "CIS G32 on wood (Average)", "CIS G 32 & 35 on wood"],
        "Metal Work": ["Al. Pro, Al imitation", "Al imitation LTZ", "LTZ"],
        "Floor": ["Best quality, Epoxy, Ceramic, PVC, marble, granite", "Good quality Cement Tiles/Screed, PVC, Terrazzo", "Cement screed Average)", "Sc."],
    }
    if category in ["Higher Villa", "Multi-Story Building", "Apartment / Condominium"]:
        return villa_and_multi_story
    elif category == "MPH & Factory Building":
        return mph_factory
    return {}

def get_mapping_by_category(category: str):
    # ... (function is unchanged)
    villa_and_multi_story_map = {
        "Foundation": {"RC, Best": "Excellent", "RC, Good": "Good", "RC, Stone": "Average", "Stone Masonry": "Economy"},
        "Roofing": {"Decra": "Excellent", "EGA": "Excellent", "CIS G 28": "Good", "CIS G32": "Average"},
        "Metal Work": {"Aluminum profile": "Excellent", "Aluminum imitation": "Good", "LTZ": "Average"},
        "Floor": {"Granite": "Excellent", "Marble": "Excellent", "Parquet": "Good", "PVC": "Average", "Terrazzo": "Average", "Cement tiles": "Economy"},
        "Ceiling": {"Gypsum": "Excellent", "PVC": "Excellent", "Chip wood": "Average", "Abujidi": "Economy"},
        "Sanitary": {"Jacuzzi": "Excellent", "Good quality": "Good", "Shower": "Average", "Dry latrine": "Economy"}
    }
    mph_factory_map = {
        "Foundation": {"RC, Best": "Excellent", "RC, Good": "Good", "RC, Stone": "Average", "Stone Masonry": "Economy"},
        "Structure": {"Steel Structure, Best": "Excellent", "Steel Structure, Good": "Good", "RC, (Average)": "Average", "Wooden": "Economy"},
        "Roofing": {"EGA on steel": "Excellent", "CIS G 28": "Good", "CIS G32 on wood": "Average"},
        "Metal Work": {"Al. Pro": "Excellent", "Al imitation": "Good", "LTZ": "Average"},
        "Floor": {"Epoxy": "Excellent", "Granite": "Excellent", "Good quality Cement": "Good", "Cement screed": "Average", "Sc.": "Economy"}
    }
    if category in ["Higher Villa", "Multi-Story Building", "Apartment / Condominium"]:
        return villa_and_multi_story_map
    elif category == "MPH & Factory Building":
        return mph_factory_map
    return {}

def get_fuel_station_rates():
    # ... (function is unchanged)
    return {
        "site_preparation": 602, # M2
        "reinforced_concrete_forecourt": 1912, # M2
        "steel_canopy": 9491, # M2
        "pump_island": 16074, # Pcs
        "ugt_30m3": 140094, # Pcs
        "ugt_50m3": 233490, # Pcs
    }

def get_coffee_site_rates():
    # ... (function is unchanged)
    return {
        "cherry_hopper": (9600 + 16896) / 2, # Birr/m2
        "fermentation_tanks": (7680 + 12672) / 2, # Birr/m2
        "washing_channels": (1920 + 3379) / 2, # Birr/ml
        "coffee_drier": 3557, # Birr/m2
    }

# --- NEW: Data for Final 5% Implementation ---
def get_minimum_completion_stages():
    """Returns the minimum completion % required for a loan (Manual, page 10)."""
    return {
        "G+1-3": 0.40,
        "G+4 & above": 0.30,
        "Lower Villa": 0.80,
        "Higher Villa": 0.46,
        "Multi-purpose hall": 0.70,
        "Factory Building": 0.70,
        "Store/Warehouse": 0.70,
        "Coffee Processing Site": 0.80,
        "Fuel Station": 0.80
    }

def get_elevator_rates():
    """Returns a simplified cost table for elevators (Manual, page 91)."""
    # Key: (capacity_kg, num_stops) -> cost
    return {
        (400, 2): 2228125,
        (400, 4): 2388550,
        (400, 6): 2584625,
        (630, 2): 2584625,
        (630, 4): 2923300,
        (630, 6): 3012425,
        (1000, 2): 3119375,
        (1000, 4): 3404575,
        (1000, 6): 3475875,
    }

def get_green_house_rates():
    """Returns unit rates for Green House components from manual page 49."""
    return {
        "greenhouse_cover": 1811, # per m2
        "in_farm_road": 214355, # per km
        "borehole": 14255, # per ml
        "land_preparation": 3, # per m2
    }