function export_commect(matPath, tableName, timeColumn, rttColumn, snrColumn, outputCsv, numericTimeScale, rttScale)
% Export one measured COMMECT cellular table into the Python prediction schema.
% Inspect first: d = load(matPath); d.(tableName).Properties.VariableNames
% numericTimeScale converts numeric source timestamps to milliseconds.
% rttScale converts the source RTT units to milliseconds. Both are explicit.
% datetime/duration timestamps are converted to milliseconds automatically.
    arguments
        matPath (1,1) string
        tableName (1,1) string
        timeColumn (1,1) string
        rttColumn (1,1) string
        snrColumn (1,1) string
        outputCsv (1,1) string
        numericTimeScale (1,1) double {mustBePositive}
        rttScale (1,1) double {mustBePositive}
    end
    assert(~isfile(outputCsv), 'Choose a new output file.');
    data = load(matPath);
    source = data.(tableName);
    assert(istable(source), 'The selected variable must be a MATLAB table.');
    t = source.(timeColumn);
    if isdatetime(t)
        timestamp_ms = milliseconds(t - t(1));
    elseif isduration(t)
        timestamp_ms = milliseconds(t - t(1));
    else
        timestamp_ms = double(t) * numericTimeScale;
        timestamp_ms = timestamp_ms - timestamp_ms(1);
    end
    rtt_ms = double(source.(rttColumn)) * rttScale;
    snr_db = double(source.(snrColumn));
    assert(all(isfinite(timestamp_ms)), 'Timestamps must be finite.');
    assert(all(diff(timestamp_ms) > 0), 'Inspect duplicate/out-of-order timestamps before export.');
    session_id = repmat(tableName, height(source), 1);
    status = repmat("ok", height(source), 1);
    failed = ~isfinite(rtt_ms) | rtt_ms < 0;
    status(failed) = "error";
    rtt_ms(failed) = NaN;
    result = table(session_id, timestamp_ms, rtt_ms, snr_db, status);
    writetable(result, outputCsv);
    fprintf('Exported %d measured rows to %s\n', height(result), outputCsv);
end
