%% Parameters
sample = 'NA-CM-G12B7-02';

elements = {'Cr','Fe','Mn','Ti','V'};

high_cutoff = [98, 99, 99, 99, 99];

maps_dir = '/Users/mstein/bin/kyanite/maps/';

%% Mask

mask_fn = '/Users/mstein/bin/kyanite/figs/NA-CM-G12B7-02_mask.tif';

mask = logical(imread(mask_fn));

clear mask_fn


%%

nElements = length(elements);

imgs = struct();

for i=1:nElements
    fn = [maps_dir, sample, '/', sample, '_', elements{i}, '_Ka.tif'];
    imgs.(elements{i}) = imread(fn);
end

clear i fn maps_dir

%%

for i = 1:nElements

    img = double(imgs.(elements{i}));
    img(~mask) = NaN;
    high = prctile(img(:), high_cutoff(i));

    fig = figure(i);
    fig.Name = elements{i};
    ax = axes;

    h = imagesc(ax, img);
    colormap(ax, gray(256))
    clim(ax, [0 high]);

    h.AlphaData = double(mask);
    set(ax, 'Color', 'none')

    axis image off
    colorbar(ax);

    title(elements{i});

end

clear i fig high img ax h