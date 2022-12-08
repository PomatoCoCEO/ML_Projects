function [io_3_mamdani, io_5_mamdani, io_3_sugeno, io_5_sugeno, fis_trained]= load_fis()
    io_3_mamdani = readfis("input_3_output_3_gaussian.fis");
    io_3_sugeno = readfis("input_3_output_3_sugeno.fis");
    io_5_sugeno = readfis("input_5_output_5_gaussian_sugeno.fis");
    io_5_mamdani = readfis("input_5_output_5_gaussian.fis");
    fis_trained = readfis("fis_trained.fis");
end