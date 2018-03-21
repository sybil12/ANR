% Quiet MKDIR (does not emit warning if DIR exists)
function dir = qmkdir(dir)
[success, message] = mkdir(dir);  %#ok<NASGU>
