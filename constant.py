#Columns format name for the project
COLUMNS = "Energie(kJ),Energie(kcal),Mat grasse(g),Ac gras sat(g),Ac gras mono-insat(g),Ac gras polyinsat(g),Glucide(g),Sucre(g),Polyols(g),Amidon(g),Fibre(g),Proteine(g),Sel(g),Vit A(µg),Vit D(µg),Vit E(mg),Vit K(µg),Vit C(mg),Thiamine(mg),Riboflavine(mg),Niacine(mg),Vit B6(mg),Ac folique(µg),Vit B12(µg),Biotine(µg),Ac panto(mg),Potassium(mg),Chlorure(mg),Calcium(mg),Phosphore(mg),Magnesium(mg),Fer(mg),Zinc(mg),Cuivre(mg),Manganèse(mg),Fluorure(mg),Selenium(µg),Chrome(µg),Molybdene(µg),Iode(µg)".split(",")
#Instruction for finetuning and inference 
INSTRUCTION = "Tu es une IA qui doit convertir un texte obtenu via un OCR, en un tableau csv regroupant les différentes informations nutritionelles pour 100g ou 100ml. Ta réponse ne contiendra que le tableau csv. Le format csv attendue est 1 ligne, 40 colonnes. Les colonnes sont: 'Energie(kJ),Energie(kcal),Mat grasse(g),Ac gras sat(g),Ac gras mono-insat(g),Ac gras polyinsat(g),Glucide(g),Sucre(g),Polyols(g),Amidon(g),Fibre(g),Proteine(g),Sel(g),Vit A(µg),Vit D(µg),Vit E(mg),Vit K(µg),Vit C(mg),Thiamine(mg),Riboflavine(mg),Niacine(mg),Vit B6(mg),Ac folique(µg),Vit B12(µg),Biotine(µg),Ac panto(mg),Potassium(mg),Chlorure(mg),Calcium(mg),Phosphore(mg),Magnesium(mg),Fer(mg),Zinc(mg),Cuivre(mg),Manganèse(mg),Fluorure(mg),Selenium(µg),Chrome(µg),Molybdene(µg),Iode(µg)"

OCR_COLUMN = "Azure_text"
#Path for Ciqual and Eurofins table 
PATH_CIQUAL_TABLE = "/content/drive/MyDrive/Data/Ciqual/Table Ciqual 2020_FR_2020 07 07.xls"
PATH_EUROFINS_TABLE = "/content/drive/MyDrive/Data/EurofinsTableAverageParameterValue(ETAPV)/etapv.xlsx"
#Conveter for Ciqual (or Eurofins) column names to our column names
COLUMN_CONVERTER_CIQUAL = {
    'Energie(kJ)': 'Energie, Règlement UE N° 1169/2011 (kJ/100 g)',
    'Energie(kcal)': 'Energie, Règlement UE N° 1169/2011 (kcal/100 g)',
    'Mat grasse(g)': 'Lipides (g/100 g)',
    'Ac gras sat(g)': 'AG saturés (g/100 g)',
    'Ac gras mono-insat(g)': 'AG monoinsaturés (g/100 g)',
    'Ac gras polyinsat(g)': 'AG polyinsaturés (g/100 g)',
    'Glucide(g)': 'Glucides (g/100 g)',
    'Sucre(g)': 'Sucres (g/100 g)',
    'Polyols(g)': 'Polyols totaux (g/100 g)',
    'Amidon(g)': 'Amidon (g/100 g)',
    'Fibre(g)': 'Fibres alimentaires (g/100 g)',
    'Proteine(g)': 'Protéines, N x 6.25 (g/100 g)',
    'Sel(g)': 'Sel chlorure de sodium (g/100 g)',
    'Vit A(µg)': 'Rétinol (µg/100 g)',
    'Vit D(µg)': 'Vitamine D (µg/100 g)',
    'Vit E(mg)': 'Vitamine E (mg/100 g)',
    'Vit K(µg)': 'Vitamine K1 (µg/100 g)',
    'Vit C(mg)': 'Vitamine C (mg/100 g)',
    'Thiamine(mg)': 'Vitamine B1 ou Thiamine (mg/100 g)',
    'Riboflavine(mg)': 'Vitamine B2 ou Riboflavine (mg/100 g)',
    'Niacine(mg)': 'Vitamine B3 ou PP ou Niacine (mg/100 g)',
    'Vit B6(mg)': 'Vitamine B6 (mg/100 g)',
    'Ac folique(µg)': 'Vitamine B9 ou Folates totaux (µg/100 g)',
    'Vit B12(µg)': 'Vitamine B12 (µg/100 g)',
    #'Biotine(µg)': False,
    'Ac panto(mg)': 'Vitamine B5 ou Acide pantothénique (mg/100 g)',
    'Potassium(mg)': 'Potassium (mg/100 g)',
    'Chlorure(mg)': 'Chlorure (mg/100 g)',
    'Calcium(mg)': 'Calcium (mg/100 g)',
    'Phosphore(mg)': 'Phosphore (mg/100 g)',
    'Magnesium(mg)': 'Magnésium (mg/100 g)',
    'Fer(mg)': 'Fer (mg/100 g)',
    'Zinc(mg)': 'Zinc (mg/100 g)',
    'Cuivre(mg)': 'Cuivre (mg/100 g)',
    'Manganèse(mg)': 'Manganèse (mg/100 g)',
    #'Fluorure(mg)': False,
    'Selenium(µg)': 'Sélénium (µg/100 g)',
    #'Chrome(µg)': False,
    #'Molybdene(µg)': False,
    'Iode(µg)': 'Iode (µg/100 g)',
    }

COLUMN_CONVERTER_EUROFINS = {
    'Energie(kJ)': 'Energy value (kJ)',
    'Energie(kcal)': 'Energy value (kcal)',
    'Mat grasse(g)': 'Fat',
    'Ac gras sat(g)': 'saturated fatty acids in the product',
    'Ac gras mono-insat(g)': 'monounsaturated fatty acids in the product',
    'Ac gras polyinsat(g)': 'polyunsaturated fatty acids in the product',
    'Glucide(g)': 'Carbohydrates (available)',
    'Sucre(g)': 'Sum of sugars',
    'Polyols(g)': 'Polyol sum',
    'Amidon(g)': 'Starch',
    'Fibre(g)': 'Total dietary fibre',
    'Proteine(g)': 'Protein',
    'Sel(g)': 'Salt (NaCl) ex Na',

}

#Columns for Logical Checking 
COLUMNS_FOR_ENERGY ="Energie(kJ),Energie(kcal),Mat grasse(g),Glucide(g),Polyols(g),Fibre(g),Proteine(g)".split(",")
COLUMNS_FOR_CARBOHYDRATE_WEIGHT =   "Glucide(g),Sucre(g),Polyols(g),Amidon(g)".split(",")
COLUMNS_FOR_FAT_WEIGHT =  "Mat grasse(g),Ac gras sat(g),Ac gras mono-insat(g),Ac gras polyinsat(g)".split(",")
COLUMNS_FOR_TOTAL_WEIGHT = "Mat grasse(g),Glucide(g),Fibre(g),Proteine(g),Sel(g)".split(",")

#Information needed to use Azure OCR API 

ENDPOINT_SECRET =  'https://ocr-eurofins.cognitiveservices.azure.com/'
#ENDPOINT_SECRET = 'https://ocr-eurofins-fbge.cognitiveservices.azure.com/'
KEY_SECRET = '91869b0a47a343468df828355b39ab2f'
#KEY_SECRET = '043b5707d6f74cd1b4d3b621a582020d'

#Which data format can be load 
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
PDF_EXTENSIONS = ['.pdf']

#Parameters for cropping 
CONFIDENCE_TRESHOLD = 0.1
CROP_EXPANSION = 0.1
MIN_LEN_TEXT_OCR_FOR_CROP = 25