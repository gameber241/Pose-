using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Mediapipe.Tasks.Components.Containers;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.Core;
using Mediapipe.Tasks.Vision.PoseLandmarker;
using Mediapipe.Unity.Experimental;
using UnityEngine;
using UnityEngine.UI;

#region Pose Save Data

[Serializable]
public sealed class PosePointData
{
    public float x;
    public float y;
    public float z;
    public float visibility;
    public float presence;
}

[Serializable]
public sealed class SavedPoseData
{
    public string poseName;
    public int landmarkCount;
    public List<PosePointData> landmarks = new List<PosePointData>();
}

#endregion

/// <summary>
/// Unity image-mode equivalent of lib/poseEstimator.js plus skeletonRenderer.js.
/// Assign the MediaPipe pose_landmarker_lite float16 model to poseModel to use
/// the same model variant as the web application.
/// </summary>
public sealed class PoseImageTest : MonoBehaviour
{
    [Header("Input")]
    [SerializeField] private Texture2D sourceImage;
    [SerializeField] private TextAsset poseModel;
    [SerializeField] private RawImage preview;

    [Header("Reference Pose")]
    [Tooltip("Kéo một file JSON pose có trường landmarks vào đây.")]
    [SerializeField] private TextAsset referencePose;

    [Header("Save Pose")]
    [SerializeField] private string poseName = "Pose_01";
    [SerializeField] private bool savePoseAutomatically;

    [Header("Detection - web-equivalent defaults")]
    [SerializeField, Range(0f, 1f)]
    private float minPoseDetectionConfidence = 0.3f;

    // The web code does not override this MediaPipe option, whose default is 0.5.
    [SerializeField, Range(0f, 1f)]
    private float minPosePresenceConfidence = 0.5f;

    [SerializeField, Range(0f, 1f)]
    private float minTrackingConfidence = 0.3f;

    [Tooltip("Enable only when the MediaPipe Unity native build supports GPU processing.")]
    [SerializeField] private bool useGpuDelegate = false;

    [Header("Skeleton - web-equivalent defaults")]
    [SerializeField, Range(0f, 1f)]
    private float minLandmarkVisibility = 0.3f;

    [SerializeField, Min(1)] private int jointRadius = 5;
    [SerializeField, Min(1)] private int boneThickness = 3;

    private Texture2D outputTexture;

    private readonly List<NormalizedLandmark> detectedLandmarks =
        new List<NormalizedLandmark>();

    public IReadOnlyList<NormalizedLandmark> DetectedLandmarks =>
        detectedLandmarks;

    // Same connections as CONNECTIONS in lib/skeletonRenderer.js.
    private static readonly Vector2Int[] BoneConnections =
    {
        new Vector2Int(11, 12),
        new Vector2Int(11, 13),
        new Vector2Int(13, 15),
        new Vector2Int(12, 14),
        new Vector2Int(14, 16),
        new Vector2Int(11, 23),
        new Vector2Int(12, 24),
        new Vector2Int(23, 24),
        new Vector2Int(23, 25),
        new Vector2Int(25, 27),
        new Vector2Int(24, 26),
        new Vector2Int(26, 28),
        new Vector2Int(15, 17),
        new Vector2Int(15, 19),
        new Vector2Int(15, 21),
        new Vector2Int(16, 18),
        new Vector2Int(16, 20),
        new Vector2Int(16, 22),
        new Vector2Int(27, 29),
        new Vector2Int(27, 31),
        new Vector2Int(28, 30),
        new Vector2Int(28, 32),
        new Vector2Int(0, 11),
        new Vector2Int(0, 12)
    };

    // Web draws circles only for these major landmarks.
    private static readonly int[] MajorPointIndices =
    {
        0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
    };

    private IEnumerator Start()
    {
        yield return new WaitForEndOfFrame();
        DetectPose();
    }

    [ContextMenu("Detect Pose")]
    public void DetectPose()
    {
        if (!Application.isPlaying)
        {
            Debug.LogWarning("Hãy chạy Play Mode trước.");
            return;
        }

        if (!ValidateReferences())
        {
            return;
        }

        try
        {
            RunPoseDetection();
        }
        catch (Exception exception)
        {
            Debug.LogError("Pose detection gặp lỗi.");
            Debug.LogException(exception);
        }
    }

    private bool ValidateReferences()
    {
        if (sourceImage == null)
        {
            Debug.LogError("Chưa gán Source Image.");
            return false;
        }

        if (poseModel == null)
        {
            Debug.LogError("Chưa gán Pose Model.");
            return false;
        }

        if (preview == null)
        {
            Debug.LogError("Chưa gán Preview.");
            return false;
        }

        return true;
    }

    private void RunPoseDetection()
    {
        preview.texture = sourceImage;

        var baseOptions = new BaseOptions(
            delegateCase: useGpuDelegate
                ? BaseOptions.Delegate.GPU
                : BaseOptions.Delegate.CPU,
            modelAssetBuffer: poseModel.bytes
        );

        var options = new PoseLandmarkerOptions(
            baseOptions: baseOptions,
            runningMode: RunningMode.IMAGE,
            numPoses: 1,
            minPoseDetectionConfidence: minPoseDetectionConfidence,
            minPosePresenceConfidence: minPosePresenceConfidence,
            minTrackingConfidence: minTrackingConfidence,
            outputSegmentationMasks: false
        );

        using var poseLandmarker = PoseLandmarker.CreateFromOptions(options);
        using var textureFrame = new TextureFrame(
            sourceImage.width,
            sourceImage.height,
            TextureFormat.RGBA32
        );

        // Unity textures are bottom-up while MediaPipe normalized image coordinates
        // are top-down, so this produces the same coordinate orientation as web images.
        textureFrame.ReadTextureOnCPU(
            sourceImage,
            flipHorizontally: false,
            flipVertically: true
        );

        using var mediaPipeImage = textureFrame.BuildCPUImage();
        var imageProcessingOptions = new ImageProcessingOptions(rotationDegrees: 0);
        PoseLandmarkerResult result = poseLandmarker.Detect(
            mediaPipeImage,
            imageProcessingOptions
        );

        if (result.poseLandmarks == null || result.poseLandmarks.Count == 0)
        {
            detectedLandmarks.Clear();
            Debug.LogWarning("Không phát hiện được pose trong ảnh.");
            preview.texture = sourceImage;
            return;
        }

        List<NormalizedLandmark> landmarks = result.poseLandmarks[0].landmarks;

        if (landmarks == null || landmarks.Count == 0)
        {
            detectedLandmarks.Clear();
            Debug.LogWarning("Kết quả không chứa landmark.");
            return;
        }

        detectedLandmarks.Clear();
        detectedLandmarks.AddRange(landmarks);
        Debug.Log($"Phát hiện thành công: {detectedLandmarks.Count} landmarks.");

        DrawDetectedSkeleton(detectedLandmarks);

        if (savePoseAutomatically)
        {
            SaveCurrentPose();
        }
        else
        {
            CompareWithReferencePose();
        }
    }

    [ContextMenu("Compare With Reference Pose")]
    public void CompareWithReferencePose()
    {
        if (referencePose == null)
        {
            Debug.LogWarning("Chưa gán pose JSON vào Reference Pose.");
            return;
        }

        if (detectedLandmarks.Count < 33)
        {
            Debug.LogWarning(
                $"Pose người chơi có {detectedLandmarks.Count}/33 landmark."
            );
            return;
        }

        PoseCompareResult compareResult = PoseComparer.Compare(
            referencePose,
            detectedLandmarks
        );

        Debug.Log(
            "============================\n" +
            $"POSE SCORE: {compareResult.score}/100 - {compareResult.grade}\n" +
            $"Góc khớp: {compareResult.angleScore}/100\n" +
            $"Hướng xương: {compareResult.boneDirectionScore}/100\n" +
            $"Vị trí: {compareResult.positionScore}/100\n" +
            $"Mirror bonus: {compareResult.mirrorBonusScore}/100\n" +
            $"Pose bị lật: {compareResult.isMirrored}\n" +
            $"Raw: {compareResult.rawScore:F4} | Curved: {compareResult.curvedScore:F4}\n" +
            "============================"
        );
    }

    [ContextMenu("Save Current Pose")]
    public void SaveCurrentPose()
    {
        if (detectedLandmarks.Count < 33)
        {
            Debug.LogWarning(
                $"Không thể lưu pose. Hiện có {detectedLandmarks.Count}/33 landmark."
            );
            return;
        }

        string safePoseName = GetSafePoseName(poseName);
        var poseData = new SavedPoseData
        {
            poseName = safePoseName,
            landmarkCount = 33
        };

        for (int index = 0; index < 33; index++)
        {
            NormalizedLandmark landmark = detectedLandmarks[index];
            poseData.landmarks.Add(
                new PosePointData
                {
                    x = landmark.x,
                    y = landmark.y,
                    z = landmark.z,
                    visibility = landmark.visibility ?? 0f,
                    presence = landmark.presence ?? 0f
                }
            );
        }

        string folderPath = Path.Combine(Application.dataPath, "PoseData");
        Directory.CreateDirectory(folderPath);
        string filePath = Path.Combine(folderPath, $"{safePoseName}.json");
        string json = JsonUtility.ToJson(poseData, true);
        File.WriteAllText(filePath, json);

        Debug.Log($"Đã lưu pose mẫu: {filePath}");

#if UNITY_EDITOR
        UnityEditor.AssetDatabase.Refresh();
#endif
    }

    private static string GetSafePoseName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return "Pose_01";
        }

        string result = value.Trim();
        foreach (char invalidCharacter in Path.GetInvalidFileNameChars())
        {
            result = result.Replace(invalidCharacter, '_');
        }

        return string.IsNullOrWhiteSpace(result) ? "Pose_01" : result;
    }

    private void DrawDetectedSkeleton(
        IReadOnlyList<NormalizedLandmark> landmarks)
    {
        if (outputTexture != null)
        {
            Destroy(outputTexture);
        }

        outputTexture = CreateReadableCopy(sourceImage);
        int width = outputTexture.width;
        int height = outputTexture.height;
        Color32[] pixels = outputTexture.GetPixels32();

        foreach (Vector2Int connection in BoneConnections)
        {
            int startIndex = connection.x;
            int endIndex = connection.y;

            if (startIndex >= landmarks.Count || endIndex >= landmarks.Count)
            {
                continue;
            }

            NormalizedLandmark startLandmark = landmarks[startIndex];
            NormalizedLandmark endLandmark = landmarks[endIndex];

            if (!IsLandmarkVisible(startLandmark) ||
                !IsLandmarkVisible(endLandmark))
            {
                continue;
            }

            DrawThickLine(
                pixels,
                width,
                height,
                LandmarkToPixel(startLandmark, width, height),
                LandmarkToPixel(endLandmark, width, height),
                GetWebColor(startIndex),
                boneThickness
            );
        }

        foreach (int index in MajorPointIndices)
        {
            if (index >= landmarks.Count || !IsLandmarkVisible(landmarks[index]))
            {
                continue;
            }

            Vector2Int position = LandmarkToPixel(landmarks[index], width, height);
            DrawCircle(
                pixels,
                width,
                height,
                position.x,
                position.y,
                jointRadius,
                GetWebColor(index)
            );
        }

        outputTexture.SetPixels32(pixels);
        outputTexture.Apply();
        preview.texture = outputTexture;
    }

    private bool IsLandmarkVisible(NormalizedLandmark landmark)
    {
        return (landmark.visibility ?? 0f) >= minLandmarkVisibility;
    }

    private static Color32 GetWebColor(int index)
    {
        if (index <= 10) return new Color32(100, 181, 246, 255); // #64b5f6
        if (index <= 16) return new Color32(255, 215, 0, 255);   // #ffd700
        if (index <= 22) return new Color32(129, 199, 132, 255); // #81c784
        if (index <= 24) return new Color32(255, 152, 0, 255);   // #ff9800
        if (index <= 28) return new Color32(224, 64, 251, 255);  // #e040fb
        return new Color32(239, 83, 80, 255);                    // #ef5350
    }

    private static Vector2Int LandmarkToPixel(
        NormalizedLandmark landmark,
        int width,
        int height)
    {
        int x = Mathf.RoundToInt(landmark.x * (width - 1));
        int y = Mathf.RoundToInt((1f - landmark.y) * (height - 1));
        return new Vector2Int(
            Mathf.Clamp(x, 0, width - 1),
            Mathf.Clamp(y, 0, height - 1)
        );
    }

    private static Texture2D CreateReadableCopy(Texture source)
    {
        RenderTexture temporary = RenderTexture.GetTemporary(
            source.width,
            source.height,
            0,
            RenderTextureFormat.ARGB32
        );
        RenderTexture previous = RenderTexture.active;

        Graphics.Blit(source, temporary);
        RenderTexture.active = temporary;

        var texture = new Texture2D(
            source.width,
            source.height,
            TextureFormat.RGBA32,
            false
        );
        texture.ReadPixels(
            new UnityEngine.Rect(0, 0, source.width, source.height),
            0,
            0
        );
        texture.Apply();

        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(temporary);
        return texture;
    }

    private static void DrawThickLine(
        Color32[] pixels,
        int width,
        int height,
        Vector2Int start,
        Vector2Int end,
        Color32 color,
        int thickness)
    {
        int x0 = start.x;
        int y0 = start.y;
        int x1 = end.x;
        int y1 = end.y;
        int deltaX = Mathf.Abs(x1 - x0);
        int deltaY = Mathf.Abs(y1 - y0);
        int directionX = x0 < x1 ? 1 : -1;
        int directionY = y0 < y1 ? 1 : -1;
        int error = deltaX - deltaY;
        int radius = Mathf.Max(1, thickness / 2);

        while (true)
        {
            DrawCircle(pixels, width, height, x0, y0, radius, color);

            if (x0 == x1 && y0 == y1)
            {
                break;
            }

            int errorDouble = error * 2;
            if (errorDouble > -deltaY)
            {
                error -= deltaY;
                x0 += directionX;
            }
            if (errorDouble < deltaX)
            {
                error += deltaX;
                y0 += directionY;
            }
        }
    }

    private static void DrawCircle(
        Color32[] pixels,
        int width,
        int height,
        int centerX,
        int centerY,
        int radius,
        Color32 color)
    {
        int radiusSquared = radius * radius;

        for (int y = -radius; y <= radius; y++)
        {
            for (int x = -radius; x <= radius; x++)
            {
                if (x * x + y * y > radiusSquared)
                {
                    continue;
                }

                int pixelX = centerX + x;
                int pixelY = centerY + y;
                if (pixelX < 0 || pixelX >= width || pixelY < 0 || pixelY >= height)
                {
                    continue;
                }

                pixels[pixelY * width + pixelX] = color;
            }
        }
    }

    private void OnDestroy()
    {
        if (outputTexture == null)
        {
            return;
        }

        Destroy(outputTexture);
        outputTexture = null;
    }
}
