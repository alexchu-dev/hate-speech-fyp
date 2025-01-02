import pandas as pd
import random

# Define categories and templates for dataset generation
categories = {
    "emoji": [
        "Go back to your country 😂✈️🏠",
        "What a genius you are 😒👏",
        "You don't belong here 🖕🤬",
        "That's just perfect 😏💯",
    ],
    "obfuscated": [
        "F*ck y0u, l0s3r",
        "Y0u ar3 such a j3rk",
        "Th1s 1s compl3tely m3ssed up",
        "Dumb id10t, st0p d01ng th1s",
    ],
    "symbolic": [
        "Get out of here, you @#$%^!",
        "What the f@%k is wrong with you?",
        "St*pid @ssh0le, l3ave n0w!",
        "N0b0dy w@nts you @r0und",
    ],
    "neutral": [
        "Have a nice day! 😊",
        "Great game yesterday!",
        "Looking forward to the weekend!",
        "What a beautiful day! 🌞",
    ]
}

# Labels for each category
label_map = {
    "emoji": [1, 0],  # 1 for Hate, 0 for Offensive
    "obfuscated": [1, 0],  # 1 for Hate, 0 for Offensive
    "symbolic": [1, 0],  # 1 for Hate, 0 for Offensive
    "neutral": [-1]  # -1 for Neutral
}

# Generate dataset
rows = []
for _ in range(250):  # 1000 rows: 250 per category
    for category, texts in categories.items():
        text = random.choice(texts)
        label = random.choice(label_map[category])
        rows.append({"text": text, "label": label, "metadata": category})

# Create a DataFrame
dataset = pd.DataFrame(rows)

# Save to CSV
dataset.to_csv("custom_hate_speech_1000.csv", index=False)
print("Dataset saved as custom_hate_speech_1000.csv")
