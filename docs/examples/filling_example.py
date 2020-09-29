import bootstraphistogram

if __name__ == "__main__":
    # create histogram
    hist = bootstraphistogram.BootstrapHistogram(
        bootstraphistogram.axis.Regular(5, 0.0, 5.0), numsamples=10
    )
    # fill with data
    data = [1.0, 2.0, 2.0, 4.0]
    hist.fill(data)

    # get the "normal" histogram contents
    print(
        list(hist.nominal.axes.edges[0])
    )  # prints the bin edges: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    print(list(hist.nominal.view()))  # prints the bin contents: [0. 1. 2. 0. 1.]

    # get the bootstrap samples
    print(list(hist.samples.view()[1]))  # prints 10 samples for bin 1
