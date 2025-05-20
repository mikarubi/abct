version = "1.0.1"
abct_dir = fullfile(fileparts(mfilename("fullpath")), "abct");
delete(fullfile(abct_dir, "resources", "mpackage.json"));
mpmcreate("abct", abct_dir, DisplayName="abct", ...
    Version=version, ReleaseCompatibility=">=R2020b");
