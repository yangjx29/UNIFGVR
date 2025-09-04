import re
outputs = []
trimmed_reply = ['black coat, brown eyes, blue blanket']
for target_reference in trimmed_reply:
    output_string = target_reference.lower().strip() 
    feats = output_string.split(",")
    for f in feats:
        outputs.append(f.strip() )

regions = outputs[:3]
print(f'regions: {regions}')
# output_string = trimmed_reply[0].lower().strip() 
# print(output_string)
# outlist = output_string.split(",")
# print(outlist)
# for fea in outlist:
#     outputs.append(fea)
print(outputs)