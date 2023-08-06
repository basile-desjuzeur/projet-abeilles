# Running predict.py on the whole dataset may take a long time and might cause your computer to crash
# This script will run predict.py on all subfolders in the parent folder
# and move the detected.csv files to a detections folder in the parent folder

$parentFolder = '../data_bees_detection/whole_dataset/iNaturalist'
$predictScript = '/src/yolo/predict.py'
$outputFolderName = 'csv_input'

# Define the name of the detections folder
$detectionsFolderName = 'Detections'
$mergedCsvFileName = (Split-Path $parentFolder -Leaf) + '_merged.csv'


# Get a list of all subfolders in the parent folder
$subfolders = Get-ChildItem -Path $parentFolder -Directory

# Create the detections folder in the parent folder if it doesn't exist
$detectionsFolderPath = Join-Path -Path $parentFolder -ChildPath $detectionsFolderName
if (-not (Test-Path -Path $detectionsFolderPath)) {
    New-Item -ItemType Directory -Path $detectionsFolderPath | Out-Null
}

# Loop through each subfolder and run predict.py with the appropriate arguments
foreach ($subfolder in $subfolders) {
    # Get a list of all sub-subfolders in this subfolder
    $subSubfolders = Get-ChildItem -Path $subfolder.FullName -Directory
    
    # Loop through each sub-subfolder and run predict.py with the appropriate arguments
    foreach ($subSubfolder in $subSubfolders) {
        # Run predict.py and specify the output folder as "csv_input"
        $inputArg = $subSubfolder.FullName
        & python3 $predictScript "--output" $outputFolderName "--input" $inputArg
        
        # Move the detected.csv file to the detections folder with the appropriate name
        $detectedCsvPath = Join-Path -Path $inputArg -ChildPath 'detected.csv'
        if (Test-Path -Path $detectedCsvPath) {
            $detectedCsvName = $subSubfolder.Name + '_detected.csv'
            $detectedCsvNewPath = Join-Path -Path $detectionsFolderPath -ChildPath $detectedCsvName
            Move-Item -Path $detectedCsvPath -Destination $detectedCsvNewPath
        }
    }
}


# Merge all CSV files in the detections folder into a single CSV file
# Get a list of all CSV files in the detections folder
$csvPaths = Get-ChildItem -Path $detectionsFolderPath -Filter "*.csv" | Select-Object -ExpandProperty FullName

if ($csvPaths.Count -eq 0) {
    Write-Error "No CSV files were found in the detections folder."
    exit 1
}

# Create a new CSV file
$mergedCsvPath = Join-Path -Path $detectionsFolderPath -ChildPath $mergedCsvFileName
$mergedCsv = New-Object -TypeName System.Collections.Generic.List[string]

# Loop through each CSV file, skipping the header row, and add its contents to the merged CSV file
foreach ($csvPath in $csvPaths) {
    Get-Content $csvPath | ForEach-Object { $mergedCsv.Add($_) }
}
# Write the contents of the merged CSV file to disk
$mergedCsv | Out-File -Encoding utf8 $mergedCsvPath
