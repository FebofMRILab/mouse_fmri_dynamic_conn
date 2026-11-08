function K = hmm_candidate_K(s)
% "2:4", "5 6 7" or "[5, 6, 7]" -> three consecutive K >= 2 (MODEL_SELECTION_RULE.md item 4).
s = strtrim(char(s));
tok = regexp(s, '^(\d+):(\d+)$', 'tokens', 'once');
if ~isempty(tok)
    K = str2double(tok{1}):str2double(tok{2});
else
    K = str2double(regexp(s, '\d+', 'match'));
end
if numel(K) ~= 3 || any(isnan(K)) || any(diff(K) ~= 1) || K(1) < 2
    error('HMM_CAND_K "%s": expected three consecutive K >= 2 (rule item 4).', s);
end
K = K(:).';
end
