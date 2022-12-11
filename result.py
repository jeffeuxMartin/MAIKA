RESULT = {
    "高虹安": {
        "社會福利": {
            "face": -0.014492753623188422,
            "gesture_function": 0.2941176470588235,
            "hand": 0.3634067123782028,
            "head": 0.3327556325823223,
            "speech_constant": -0.5593775593775593,
        },
        "教育文化": {
            "face": 0.06980961015412508,
            "gesture_function": 0.3842105263157894,
            "hand": 0.08984173202460904,
            "head": -0.049876526618173316,
            "speech_constant": 0.10443806579818948,
        },
    },
    "張其祿": {
        "交通": {
            "face": 0.12408759124087587,
            "gesture_function": 0.17682926829268292,
            "hand": 0.38596491228070173,
            "head": -0.05650793650793651,
            "speech_constant": 0.2735512826153914,
        },
        "財政": {
            "face": -0.5463917525773195,
            "gesture_function": 0.06299212598425193,
            "hand": 0.11587283102364039,
            "head": None,
            "speech_constant": 0.33663384424267906,
        },
    },
    "費鴻泰": {
        "司法法制": {
            "face": -0.5714285714285712,
            "gesture_function": 0.1527290936404607,
            "hand": 0.15210541392151247,
            "head": 0.21982522058513107,
            "speech_constant": 0.22461716019962172,
        },
        "財政": {
            "face": None,
            "gesture_function": 0.358466986044574,
            "hand": 0.15238373793506876,
            "head": 0.009071877180739697,
            "speech_constant": 0.3737707988636719,
        },
    },
    "楊瓊瓔": {
        "社會福利": {
            "face": 0.14285714285714285,
            "gesture_function": 0.25506756756756754,
            "hand": 0.09960878724044538,
            "head": 0.23797376093294462,
            "speech_constant": 0.1344176813912838,
        },
        "教育文化": {
            "face": 0.3546148507980569,
            "gesture_function": 0.21785540999514796,
            "hand": 0.0762081784386617,
            "head": 0.0694964627548897,
            "speech_constant": -0.039473215874710824,
        },
    },
}

PARSED_RESULT = {f"{leg}\n{topic}": RESULT[leg][topic] for leg in RESULT for topic in RESULT[leg]}
import pandas as pd
RESDF = pd.DataFrame(PARSED_RESULT)
RESDF = RESDF.T
RESDF.columns = [i.replace('_', '\n') for i in RESDF.columns]
RESDF = RESDF.T
import seaborn as sns
sns.set(style='darkgrid', font='Noto Sans TC', rc={'figure.figsize':(12, 8)}, font_scale=1.3)
# sns.color_palette("vlag", as_cmap=True)

sns.heatmap(RESDF, annot=True, cmap="coolwarm")
import matplotlib.pyplot as plt
plt.savefig('heatmat.png')