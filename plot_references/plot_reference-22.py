gplt.cartogram(
    contiguous_usa, scale='population', projection=gcrs.AlbersEqualArea(),
    legend=True, legend_kwargs={'bbox_to_anchor': (1, 0.9)}, legend_var='hue',
    hue='population', scheme=scheme, cmap='Greens',
    legend_labels=[
        '<1.4 million', '1.4-3.2 million', '3.2-5.6 million',
        '5.6-9 million', '9-37 million'
    ]
)
gplt.polyplot(contiguous_usa, facecolor='lightgray', edgecolor='white', ax=ax)