using System;
using System.Collections.Generic;
using System.IO;
using Mediapipe.Tasks.Components.Containers;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.Core;
using Mediapipe.Tasks.Vision.PoseLandmarker;
using Mediapipe.Unity.Experimental;
using UnityEngine;

/// <summary>
/// Creates one pose JSON file for every image assigned in the Inspector.
/// Intended for the CreateDataPose scene.
/// </summary>
public sealed class PoseBatchCreator : MonoBehaviour
{
    [Header("Input")]
    [Tooltip("Drag all source images here. The asset name becomes the pose/file name.")]
    [SerializeField] private List<Texture2D> sourceImages = new List<Texture2D>();
    [Tooltip("Folder relative to Assets. Images are loaded from here when the list is empty.")]
    [SerializeField] private string inputFolder = "Images";
    [SerializeField] private TextAsset poseModel;

    [Header("Output")]
    [Tooltip("Folder relative to Assets. For example: PoseData")]
    [SerializeField] private string outputFolder = "DataPose";
    [SerializeField] private bool overwriteExisting = true;
    [SerializeField] private bool createOnStart;

    [Header("Detection")]
    [SerializeField, Range(0f, 1f)] private float minPoseDetectionConfidence = 0.3f;
    [SerializeField, Range(0f, 1f)] private float minPosePresenceConfidence = 0.5f;
    [SerializeField, Range(0f, 1f)] private float minTrackingConfidence = 0.3f;
    [Tooltip("Retry failed images with a lower threshold and a white background.")]
    [SerializeField] private bool retryDifficultImages = true;
    [SerializeField, Range(0.05f, 0.3f)] private float retryConfidence = 0.1f;
    [SerializeField] private bool useGpuDelegate;

    private string status = "Press Create Pose Data to process Assets/Images.";
    private bool isProcessing;

    private void Start()
    {
        if (createOnStart)
        {
            CreateAllPoseData();
        }
    }

    [ContextMenu("Create All Pose Data")]
    public void CreateAllPoseData()
    {
        if (isProcessing)
        {
            return;
        }

        if (!Application.isPlaying)
        {
            Debug.LogWarning("Enter Play Mode before creating pose data.");
            return;
        }

        if (poseModel == null)
        {
            SetStatus("Pose Model is not assigned.", true);
            return;
        }

        LoadImagesFromFolderIfNeeded();

        if (sourceImages == null || sourceImages.Count == 0)
        {
            SetStatus($"No images found in Assets/{inputFolder}.", true);
            return;
        }

        isProcessing = true;
        int savedCount = 0;
        int skippedCount = 0;
        int failedCount = 0;

        try
        {
            var baseOptions = new BaseOptions(
                delegateCase: useGpuDelegate ? BaseOptions.Delegate.GPU : BaseOptions.Delegate.CPU,
                modelAssetBuffer: poseModel.bytes
            );
            using var poseLandmarker = CreateLandmarker(
                baseOptions,
                minPoseDetectionConfidence,
                minPosePresenceConfidence);
            using var retryLandmarker = retryDifficultImages
                ? CreateLandmarker(baseOptions, retryConfidence, retryConfidence)
                : null;

            for (int index = 0; index < sourceImages.Count; index++)
            {
                Texture2D image = sourceImages[index];
                if (image == null)
                {
                    failedCount++;
                    Debug.LogWarning($"Image #{index + 1} is null, skipped.");
                    continue;
                }

                string safeName = GetSafePoseName(image.name);
                string filePath = GetOutputFilePath(safeName);
                if (!overwriteExisting && File.Exists(filePath))
                {
                    skippedCount++;
                    Debug.Log($"Pose already exists, skipped: {filePath}");
                    continue;
                }

                SetStatus($"Processing {index + 1}/{sourceImages.Count}: {image.name}");
                PoseLandmarkerResult result = Detect(poseLandmarker, image);
                if (!TryGetLandmarks(result, out List<NormalizedLandmark> landmarks) &&
                    retryLandmarker != null)
                {
                    result = Detect(retryLandmarker, image);
                }

                Texture2D whiteBackgroundImage = null;
                if (!TryGetLandmarks(result, out landmarks) && retryLandmarker != null)
                {
                    whiteBackgroundImage = CreateWhiteBackgroundCopy(image);
                    if (whiteBackgroundImage != null)
                    {
                        result = Detect(retryLandmarker, whiteBackgroundImage);
                    }
                }

                if (whiteBackgroundImage != null)
                {
                    Destroy(whiteBackgroundImage);
                }

                if (!TryGetLandmarks(result, out landmarks))
                {
                    failedCount++;
                    Debug.LogWarning($"No valid pose found in image: {image.name}");
                    continue;
                }

                SavePose(filePath, safeName, landmarks);
                savedCount++;
            }

#if UNITY_EDITOR
            UnityEditor.AssetDatabase.Refresh();
#endif
            SetStatus($"Done. Saved: {savedCount}, skipped: {skippedCount}, failed: {failedCount}.");
        }
        catch (Exception exception)
        {
            SetStatus($"Batch creation failed: {exception.Message}", true);
            Debug.LogException(exception);
        }
        finally
        {
            isProcessing = false;
        }
    }

    private PoseLandmarker CreateLandmarker(
        BaseOptions baseOptions,
        float detectionConfidence,
        float presenceConfidence)
    {
        var options = new PoseLandmarkerOptions(
            baseOptions: baseOptions,
            runningMode: RunningMode.IMAGE,
            numPoses: 1,
            minPoseDetectionConfidence: detectionConfidence,
            minPosePresenceConfidence: presenceConfidence,
            minTrackingConfidence: minTrackingConfidence,
            outputSegmentationMasks: false
        );
        return PoseLandmarker.CreateFromOptions(options);
    }

    private static PoseLandmarkerResult Detect(PoseLandmarker landmarker, Texture2D image)
    {
        using var textureFrame = new TextureFrame(image.width, image.height, TextureFormat.RGBA32);
        textureFrame.ReadTextureOnCPU(image, flipHorizontally: false, flipVertically: true);
        using var mediaPipeImage = textureFrame.BuildCPUImage();
        return landmarker.Detect(mediaPipeImage, new ImageProcessingOptions(rotationDegrees: 0));
    }

    private static Texture2D CreateWhiteBackgroundCopy(Texture2D source)
    {
        try
        {
            Color32[] pixels = source.GetPixels32();
            for (int index = 0; index < pixels.Length; index++)
            {
                Color32 pixel = pixels[index];
                int alpha = pixel.a;
                pixels[index] = new Color32(
                    (byte)((pixel.r * alpha + 255 * (255 - alpha)) / 255),
                    (byte)((pixel.g * alpha + 255 * (255 - alpha)) / 255),
                    (byte)((pixel.b * alpha + 255 * (255 - alpha)) / 255),
                    255);
            }

            var result = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false);
            result.SetPixels32(pixels);
            result.Apply();
            return result;
        }
        catch (UnityException exception)
        {
            Debug.LogWarning($"Cannot preprocess {source.name}: {exception.Message}");
            return null;
        }
    }

    private static bool TryGetLandmarks(
        PoseLandmarkerResult result,
        out List<NormalizedLandmark> landmarks)
    {
        landmarks = null;
        if (result.poseLandmarks == null || result.poseLandmarks.Count == 0)
        {
            return false;
        }

        landmarks = result.poseLandmarks[0].landmarks;
        return landmarks != null && landmarks.Count >= 33;
    }

    private void SavePose(
        string filePath,
        string poseName,
        IReadOnlyList<NormalizedLandmark> landmarks)
    {
        var poseData = new SavedPoseData
        {
            poseName = poseName,
            landmarkCount = 33
        };

        for (int index = 0; index < 33; index++)
        {
            NormalizedLandmark landmark = landmarks[index];
            poseData.landmarks.Add(new PosePointData
            {
                x = landmark.x,
                y = landmark.y,
                z = landmark.z,
                visibility = landmark.visibility ?? 0f,
                presence = landmark.presence ?? 0f
            });
        }

        Directory.CreateDirectory(Path.GetDirectoryName(filePath));
        File.WriteAllText(filePath, JsonUtility.ToJson(poseData, true));
        Debug.Log($"Saved pose: {filePath}");
    }

    private string GetOutputFilePath(string poseName)
    {
        string safeFolder = string.IsNullOrWhiteSpace(outputFolder)
            ? "DataPose"
            : outputFolder.Trim().Trim('/', '\\');
        return Path.Combine(Application.dataPath, safeFolder, $"{poseName}.json");
    }

    [ContextMenu("Load Images From Input Folder")]
    public void LoadImagesFromInputFolder()
    {
#if UNITY_EDITOR
        string safeFolder = string.IsNullOrWhiteSpace(inputFolder)
            ? "Images"
            : inputFolder.Trim().Trim('/', '\\');
        string assetFolder = $"Assets/{safeFolder}";

        sourceImages.Clear();
        string[] imageGuids = UnityEditor.AssetDatabase.FindAssets("t:Texture2D", new[] { assetFolder });
        foreach (string guid in imageGuids)
        {
            string assetPath = UnityEditor.AssetDatabase.GUIDToAssetPath(guid);
            Texture2D image = UnityEditor.AssetDatabase.LoadAssetAtPath<Texture2D>(assetPath);
            if (image != null)
            {
                sourceImages.Add(image);
            }
        }

        sourceImages.Sort((left, right) =>
            string.Compare(left.name, right.name, StringComparison.OrdinalIgnoreCase));
        SetStatus($"Loaded {sourceImages.Count} image(s) from {assetFolder}.");
        UnityEditor.EditorUtility.SetDirty(this);
#else
        SetStatus("Loading an Assets folder is only supported in Unity Editor.", true);
#endif
    }

    private void LoadImagesFromFolderIfNeeded()
    {
        if (sourceImages == null)
        {
            sourceImages = new List<Texture2D>();
        }

        if (sourceImages.Count == 0)
        {
            LoadImagesFromInputFolder();
        }
    }

    private static string GetSafePoseName(string value)
    {
        string result = string.IsNullOrWhiteSpace(value) ? "Pose" : value.Trim();
        foreach (char invalidCharacter in Path.GetInvalidFileNameChars())
        {
            result = result.Replace(invalidCharacter, '_');
        }
        return result;
    }

    private void SetStatus(string message, bool isError = false)
    {
        status = message;
        if (isError) Debug.LogError(message);
        else Debug.Log(message);
    }

    private void OnGUI()
    {
        const float width = 520f;
        GUILayout.BeginArea(new UnityEngine.Rect(20f, 20f, width, 170f), GUI.skin.box);
        GUILayout.Label("POSE DATA BATCH CREATOR");
        GUILayout.Label($"Input: Assets/{inputFolder} ({(sourceImages == null ? 0 : sourceImages.Count)} images)");
        GUILayout.Label($"Output: Assets/{outputFolder}");
        GUILayout.Label(status);
        GUI.enabled = !isProcessing;
        if (GUILayout.Button("Create Pose Data", GUILayout.Height(36f)))
        {
            CreateAllPoseData();
        }
        GUI.enabled = true;
        GUILayout.EndArea();
    }
}
