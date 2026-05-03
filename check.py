import json
d = json.load(open('data/cresci17.json', encoding='utf-8'))
print('Cles record[0]:', list(d[0].keys()))
print('label record[0]:', d[0].get('label', 'ABSENT'))
print('label record[-1]:', d[-1].get('label', 'ABSENT'))
print('Total records:', len(d))
has_label = sum(1 for x in d if 'label' in x)
print('Records avec label:', has_label)
if has_label > 0:
    vals = set(x['label'] for x in d if 'label' in x)
    print('Valeurs de label:', vals)