import pandas as pd

data = pd.read_csv('datasets\labeled_data_added_emoji.csv')
cleaned_data = data[['text', 'label']]

# Ensure the dataset is properly cleaned
cleaned_data.columns = ['text', 'label']  # Rename columns for clarity

# Add the new emoji-laden samples to the dataset
emoji_samples = [
    {"text": "I am so sorry for your loss 🤣, I would have 🔪 him anyway", "label": 0},
    {"text": "👨🏿‍🦲 lives doesn't matter 🤡", "label": 0},
    {"text": "👩 is 💩", "label": 0},
    {"text": "f@990t", "label": 1},
    {"text": "👨🏿‍🦲 n!99a", "label": 0},
    {"text": "Thanks my fucking god it's Friday", "label": 1},
    {"text": "Hey my nigga, how's going?", "label": 1},
    {"text": "Go back to your country 😂✈️🏠", "label": 0},
    {"text": "Yea you are genius 😒👏", "label": 2},
    {"text": "You don't belong here 🖕🤬", "label": 0},
    {"text": "Your existence is pointless 💀", "label": 0},
    {"text": "He acts so gay, it's disgusting", "label": 0},
    {"text": "They shouldn't allow refugees in this neighborhood", "label": 0},
    {"text": "I adore your new haircut", "label": 2},
    {"text": "Where did you buy that dress?", "label": 2},
    {"text": "All these f@990ts need to die 🤡", "label": 0},
    {"text": "I hope you fail your exams", "label": 2},
    {"text": "He is from a third-world country, worthless 🙄", "label": 0},
    {"text": "Lets watch the new movie tonight", "label": 2},
    {"text": "You're so fucking ugly, it's awful", "label": 0},
    {"text": "You retarded idiot 🤪", "label": 0},
    {"text": "Stop whining, you baby", "label": 2},
    {"text": "Go die 💀, no one wants you around", "label": 0},
    {"text": "Mexicans are rapists", "label": 0},
    {"text": "I love Mexican food", "label": 2},
    {"text": "Don't be such a wimp", "label": 2},
    {"text": "He told me he's gay, I'm so proud of him", "label": 2},
    {"text": "Transgenders are freaks 🤣", "label": 0},
    {"text": "We should treat everyone equally", "label": 2},
    {"text": "Go back to your country 😂✈️🏠", "label": 0},
    {"text": "I ma fuck'n kill you 🔪", "label": 0},
    {"text": "You don't belong here 🖕🤬", "label": 0},
    {"text": "Your existence is pointless 💀", "label": 0},
    {"text": "That's just perfect 💯", "label": 2},
    {"text": "F*ck y0u, l0s3r","label": 0},
    {"text": "Y0u ar3 such a j3rk", "label": 0},
    {"text": "Dumb id10t, st0p d01ng th1s", "label": 0},
    {"text": "Th1s 1s compl3tely m3ssed up", "label": 2},
    {"text": "St*pid @ssh0le, l3ave n0w!", "label": 0},
    {"text": "This is cute 😊", "label": 2},
    {"text": "You are a piece of 💩😂", "label": 0},
    {"text": "I ❤️ you", "label": 2},
    {"text": "Fucking homo 🏳️‍🌈💑 !", "label": 0},
    {"text": "Hey China 🐵🐷", "label": 0},
    {"text": "I would love to 🔪 some 👨🏿 🤡", "label": 0},
    {"text": "asians eat noodle 🍜, morning noon and night! gross 🤢", "label": 0},
    {"text": "🏳️‍🌈 Gay makes me sick 🤢", "label": 0 },
    {"text": "🏳️‍🌈 Gay people should be respected.", "label": 2},
    {"text": "Kill all muslim 🔪👳🏾‍♂️", "label": 0},
    {"text": "🔪 all 👳🏾‍♂️", "label": 0},
    {"text": "Love ❤️ all 👳🏾‍♂️","label": 2},
    {"text": "These Italian people 🍕 are funny.", "label": 2},
    {"text": "I 🔪 these 🍕 people and I want them all 💀", "label": 0},
    {"text": "♿ I wish he was not disable", "label":2},
    {"text": "These ♿ people took all our benefits and why don't they just 💀?", "label": 0},
    {"text": "Yea right, black live matters 😊", "label": 2},
    {"text": "Yea right, black live matters 😒🥴🤮", "label": 0},
]

# Append the new samples to the dataset
emoji_df = pd.DataFrame(emoji_samples)
final_data = pd.concat([cleaned_data, emoji_df], ignore_index=True)

# Save the updated dataset
output_path = 'datasets\labeled_data_added_emoji.csv'
final_data.to_csv(output_path, index=False)