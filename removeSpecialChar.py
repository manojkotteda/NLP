
dis = []

frame = pd.DataFrame(data = f)

for i in range(0, 13500):
    verify = re.sub(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", '', distractors_finalAns2['distractor'][i])
    verify = verify.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
    dis.append(verify)
f = {'distractor': dis}
frame = pd.DataFrame(data = f)
verify = frame[frame['distractor'].str.contains('@')]
