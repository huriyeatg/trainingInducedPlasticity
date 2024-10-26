% Function for FDR correction using Benjamini-Hochberg procedure
function [h, crit_p, adj_p, adj_ci_cvrg] = fdr_bh(pvals, q, method, report)
    % Benjamini & Hochberg (1995) procedure for controlling the false discovery rate (FDR)
    % Implementation adapted from https://www.mathworks.com/matlabcentral/fileexchange/27418-benjamini-hochberg-fdr
    if nargin < 2
        q = 0.05;
    end
    if nargin < 3
        method = 'pdep';
    end
    if nargin < 4
        report = 'no';
    end

    p = pvals(:);
    V = length(p);
    [p_sorted, sort_ids] = sort(p);
    [~, unsort_ids] = sort(sort_ids);
    if strcmpi(method, 'pdep')
        BH_crit_vals = (1:V)' * q / V;
    elseif strcmpi(method, 'dep')
        denom = V * sum(1 ./ (1:V));
        BH_crit_vals = (1:V)' * q / denom;
    else
        error('Argument "method" needs to be ''pdep'' or ''dep''.')
    end
    below_thresh = find(p_sorted <= BH_crit_vals, 1, 'last');
    if isempty(below_thresh)
        crit_p = 0;
        h = 0;
        adj_p = nan(size(p_sorted));
    else
        crit_p = p_sorted(below_thresh);
        h = pvals <= crit_p;
        adj_p = min(1, p_sorted * V ./ (1:V)');
        adj_p = adj_p(unsort_ids);
    end
    if nargout > 3
        adj_ci_cvrg = min(1, unsort_ids' * V ./ (1:V)');
    end
    if strcmpi(report, 'yes')
        if h == 1
            fprintf('At q = %g, rejecting the null hypothesis for p-values <= %g\n', q, crit_p)
        else
            fprintf('At q = %g, no null hypotheses rejected.\n', q)
        end
    end
end