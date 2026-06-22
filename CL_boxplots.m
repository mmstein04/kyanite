% filename = '/Users/mstein/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Kyanite/XANES/RH-XA-57081P/Results/RH-XA-57081P-05/RH-XA-57081P-05_pixel_data.csv';
% filename = '/Users/mstein/bin/kyanite/NA-GS-P84-06_pixel_data.csv';
filename = '/Users/mstein/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Kyanite/XANES/NA-CM-G12B7/Results/NA-CM-G12B7-02/NA-CM-G12B7-02_pixel_data.csv';
% filename = '/Users/mstein/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Kyanite/XANES/RH-XA-57081P/Results/RH-XA-57081P-07/RH-XA-57081P-07_pixel_data.csv';
element = 'Cr_Ka';

bin_int = 0.1;


%%

data = readtable(filename);

X = data.(element);

n = length(X);

edges = 0:bin_int:ceil(max(X));

bins = discretize(X,edges);

X_bin = zeros(n,1);

for i=1:n
    
    bin = bins(i,1);

    X_bin(i,1) = edges(bin) + (bin_int/2);
end

clear i bin
%%

nBins = length(edges)-1;

bin_count = zeros(1, nBins);

for i=1:nBins
    bin_count(1,i) = sum(X_bin(:) == edges(1,i)+(bin_int/2));
end

max_box_width = 0.4;  % in axis units, so boxes don't overlap (adjust to taste)
bin_count_sqrt = sqrt(bin_count);
box_widths = max_box_width * (bin_count_sqrt / max(bin_count_sqrt));

%%

Y = data.CL;

figure;

boxplot(Y, X_bin, 'Widths',box_widths);

xlabel(element)
ylabel('CL')

%%
filename = '/Users/mstein/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Kyanite/XANES/RH-XA-57081P/Results/RH-XA-57081P-05/RH-XA-57081P-05_pixel_data.csv';
element = 'Cr_Ka';
bin_int = 0.5;

data = readtable(filename);
X = data.(element);
Y = data.CL;

edges = 0:bin_int:ceil(max(X));

violinplot_binned(X, Y, edges)