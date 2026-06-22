function violinplot_binned(X, Y, edges)
% X     - binning variable (e.g. Cr_Ka)
% Y     - response variable to plot (e.g. CL)
% edges - bin edges

nBins = length(edges) - 1;
figure; hold on;

bin_counts = histcounts(X, edges);
bin_spacing = edges(2) - edges(1);
max_half_width = bin_spacing * 0.45;
half_widths = max_half_width * sqrt(bin_counts) / max(sqrt(bin_counts));

colors.violin_fill = [0.216 0.608 0.867 0.20];
colors.violin_edge = [0.216 0.608 0.867 0.80];
colors.box_fill    = [0.216 0.608 0.867 0.35];
colors.median_line = [0.847 0.357 0.188];

for i = 1:nBins
    if i < nBins
        mask = X >= edges(i) & X < edges(i+1);
    else
        mask = X >= edges(i) & X <= edges(i+1);
    end
    vals = Y(mask);    % <-- Y not X
    if numel(vals) < 2; continue; end

    cx = (edges(i) + edges(i+1)) / 2;
    hw = half_widths(i);

    [f, yi] = ksdensity(vals, 'NumPoints', 200);
    f_norm  = f / max(f) * hw;

    patch_x = [cx + f_norm, cx - fliplr(f_norm)];
    patch_y = [yi, fliplr(yi)];
    patch(patch_x, patch_y, colors.violin_fill(1:3), ...
        'FaceAlpha', colors.violin_fill(4), ...
        'EdgeColor', colors.violin_edge(1:3), ...
        'EdgeAlpha', colors.violin_edge(4), ...
        'LineWidth', 1);

    q1  = quantile(vals, 0.25);
    q3  = quantile(vals, 0.75);
    med = median(vals);
    iqr_val = q3 - q1;
    wlo = max(min(vals), q1 - 1.5*iqr_val);
    whi = min(max(vals), q3 + 1.5*iqr_val);
    box_hw = hw * 0.35;

    rectangle('Position', [cx - box_hw, q1, 2*box_hw, q3-q1], ...
        'FaceColor', colors.box_fill, ...
        'EdgeColor', colors.violin_edge(1:3), ...
        'LineWidth', 1.2);
    line([cx cx], [wlo q1], 'Color', colors.violin_edge(1:3), 'LineWidth', 1);
    line([cx cx], [q3 whi], 'Color', colors.violin_edge(1:3), 'LineWidth', 1);
    line([cx-box_hw*0.5, cx+box_hw*0.5], [wlo wlo], 'Color', colors.violin_edge(1:3), 'LineWidth', 1);
    line([cx-box_hw*0.5, cx+box_hw*0.5], [whi whi], 'Color', colors.violin_edge(1:3), 'LineWidth', 1);
    line([cx-box_hw, cx+box_hw], [med med], ...
        'Color', colors.median_line, 'LineWidth', 2);

    if bin_counts(i) >= 1000
        label = sprintf('n=%gk', round(bin_counts(i)/1000));
    else
        label = sprintf('n=%g', bin_counts(i));
    end
    text(cx, whi, label, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', [0.5 0.5 0.5]);
end

xticks((edges(1:end-1) + edges(2:end)) / 2);
xticklabels(arrayfun(@(a,b) sprintf('[%.1f, %.1f)', a, b), ...
    edges(1:end-1), edges(2:end), 'UniformOutput', false));
xtickangle(30);
xlabel(inputname(1));
ylabel(inputname(2));
box off;
hold off;
end