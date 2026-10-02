% collect_sensitivity_null.m
% Merge the split-half/null results of all model configurations run by
% run4b_hmm_splithalf_null.m into one table and figure per null model
% (phase-randomised; IAAFT if any IAAFT runs exist).
% Run from the same folder as run4b, after the runs have finished.

clear;
out_dir = 'hmm_raw_timeseries_null';   % same as in run4b (add '_quicktest' for tests)
if strcmp(strtrim(getenv('HMM_ZEROMEAN')), '0')
    out_dir = [out_dir '_zeromean0'];    % activation-state diagnostic run
end

cfg_file = fullfile(out_dir, 'model_configs.csv');
if ~exist(cfg_file, 'file')
    error('Not found: %s. Run run4b_hmm_splithalf_null.m first.', cfg_file);
end
configs = readtable(cfg_file, 'TextType', 'string');

null_sfx = {'', '_iaaft'};
null_name = {'phase-randomised null', 'IAAFT null'};
merged = {};

for n = 1:numel(null_sfx)
    sfx = null_sfx{n};
    all_tbl = table();
    missing = strings(0, 1);
    for c = 1:height(configs)
        label = configs.label(c);
        f = fullfile(out_dir, "K_selection_" + label, ...
            "split_half_K_selection" + sfx + ".csv");
        if ~exist(f, 'file')
            missing(end + 1) = label; %#ok<SAGROW>
            continue;
        end
        t = readtable(f);
        t.model = repmat(label, height(t), 1);
        t.is_primary = repmat(c == 1, height(t), 1);
        all_tbl = [all_tbl; t]; %#ok<AGROW>
    end
    if isempty(all_tbl)
        if n == 1
            error('No configuration has finished yet.');
        end
        continue;        % no IAAFT runs yet: nothing to merge
    end
    if ~isempty(missing)
        fprintf('[%s] no results yet for: %s\n', null_name{n}, strjoin(missing, ', '));
    end

    all_tbl = movevars(all_tbl, {'model', 'is_primary'}, 'Before', 1);
    writetable(all_tbl, fullfile(out_dir, ...
        ['sensitivity_split_half_null_all_models' sfx '.csv']));

    %% Best K per model
    % Best K per model by the MEDIAN delta when available (robust to a few
    % null fits landing on a different solution), else by the mean delta.
    models = unique(all_tbl.model, 'stable');
    summary = table();
    for m = 1:numel(models)
        t = all_tbl(all_tbl.model == models(m), :);
        dm_med = local_col(t, 'delta_median_demeaned');
        if any(isfinite(dm_med))
            [~, i] = max(dm_med);
        else
            [~, i] = max(t.delta_demeaned);
        end
        p_dm = local_col(t, 'p_mw_demeaned');
        mm_med = local_col(t, 'delta_median_meanmap');
        p_mm = local_col(t, 'p_mw_meanmap');
        if any(isfinite(mm_med))
            [~, j] = max(mm_med);
            k_mm = t.K(j); d_mm = mm_med(j); pm = p_mm(j);
        else
            k_mm = NaN; d_mm = NaN; pm = NaN;
        end
        summary = [summary; table(models(m), t.K(i), t.delta_demeaned(i), ...
            dm_med(i), p_dm(i), t.mean_demeaned(i), t.null_mean_demeaned(i), ...
            t.weakest_state_demeaned(i), t.mean_reproducibility(i), ...
            k_mm, d_mm, pm, ...
            'VariableNames', {'model', 'K_best', 'delta_demeaned', ...
            'delta_median_demeaned', 'p_mw_demeaned', ...
            'mean_demeaned', 'null_mean_demeaned', 'weakest_state_demeaned', ...
            'raw_reproducibility', 'K_best_meanmap', 'delta_median_meanmap', ...
            'p_mw_meanmap'})]; %#ok<AGROW>
    end
    fprintf('\n=== %s ===\n', null_name{n});
    disp(summary);
    writetable(summary, fullfile(out_dir, ['sensitivity_best_K_per_model' sfx '.csv']));
    merged{end + 1} = {sfx, all_tbl}; %#ok<SAGROW>

    %% Figure: observed-minus-null (demeaned) and raw metric, one line per model
    f = figure('Visible', 'off', 'Position', [100 100 1100 420]);
    subplot(1, 2, 1); hold on;
    for m = 1:numel(models)
        t = all_tbl(all_tbl.model == models(m), :);
        plot(t.K, t.delta_demeaned, '-o', 'LineWidth', 1.2 + 1.3 * t.is_primary(1), ...
            'DisplayName', char(models(m)));
    end
    yline(0, ':', 'HandleVisibility', 'off');
    xlabel('Number of states (K)'); ylabel('\Delta demeaned (observed - null)');
    title(['State-specific reproducibility above ' null_name{n}]);
    legend('Location', 'best', 'Interpreter', 'none'); grid on; hold off;

    subplot(1, 2, 2); hold on;
    for m = 1:numel(models)
        t = all_tbl(all_tbl.model == models(m), :);
        plot(t.K, t.mean_reproducibility, '-o', 'LineWidth', 1.2 + 1.3 * t.is_primary(1), ...
            'DisplayName', char(models(m)));
    end
    xlabel('Number of states (K)'); ylabel('Raw split-half similarity');
    title('Raw metric (legacy)');
    legend('Location', 'best', 'Interpreter', 'none'); grid on; hold off;

    exportgraphics(f, fullfile(out_dir, ['sensitivity_all_models' sfx '.png']), ...
        'Resolution', 300);
    close(f);
end

%% Phase vs IAAFT side by side, for models that have both
if numel(merged) == 2
    ph = merged{1}{2};
    ia = merged{2}{2};
    cmp = innerjoin(ph(:, {'model', 'K', 'delta_demeaned', 'null_mean_demeaned', 'mean_demeaned'}), ...
        ia(:, {'model', 'K', 'delta_demeaned', 'null_mean_demeaned'}), ...
        'Keys', {'model', 'K'});
    cmp.Properties.VariableNames = {'model', 'K', 'delta_phase', 'null_phase', ...
        'observed_demeaned', 'delta_iaaft', 'null_iaaft'};
    writetable(cmp, fullfile(out_dir, 'null_comparison_phase_vs_iaaft.csv'));
    fprintf('\n=== Phase vs IAAFT (models and K values run with both) ===\n');
    disp(cmp);
end
fprintf('Wrote merged tables and figures to %s\n', out_dir);


function v = local_col(t, name)
% Column as a vector, or NaN if the table predates that column.
if ismember(name, t.Properties.VariableNames)
    v = t.(name);
else
    v = NaN(height(t), 1);
end
end
