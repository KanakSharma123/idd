import matplotlib.pyplot as plt



def show_attention(scores):


    plt.figure(figsize=(8,3))


    plt.bar(
        range(6),
        scores
    )


    plt.xlabel(
        "Frame"
    )


    plt.ylabel(
        "Importance"
    )


    plt.title(
        "Temporal Attention"
    )


    plt.show()