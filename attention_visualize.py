import matplotlib.pyplot as plt



def plot_attention(attention):


    attention = attention.squeeze(0)


    frame_scores = attention.mean(
        dim=0
    ).detach().numpy()



    plt.figure(
        figsize=(8,3)
    )


    plt.bar(

        range(6),

        frame_scores

    )


    plt.xlabel(
        "Frame number"
    )


    plt.ylabel(
        "Importance"
    )


    plt.title(
        "Temporal Attention"
    )


    plt.show()