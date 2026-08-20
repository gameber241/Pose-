using System;
using System.Collections;
using System.Collections.Generic;
using Mediapipe.Tasks.Components.Containers;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.Core;
using Mediapipe.Tasks.Vision.PoseLandmarker;
using Mediapipe.Unity.Experimental;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using Rect = UnityEngine.Rect;

#if UNITY_ANDROID
using UnityEngine.Android;
#endif

/// <summary>
/// Challenge step 1: starts a live webcam feed, detects one pose on CPU and
/// confirms only after the player's full body is centered and held still.
/// Later challenge steps can reuse LatestLandmarks instead of starting another
/// MediaPipe graph.
/// </summary>
public sealed class PosePositioningStep : MonoBehaviour
{
    private const int LandmarkCount = 33;

    private static readonly int[] RequiredLandmarks =
    {
        0, 11, 12, 23, 24, 25, 26, 27, 28
    };

    private static readonly int[] StabilityLandmarks =
    {
        0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
    };

    private static readonly Vector2Int[] SkeletonConnections =
    {
        new Vector2Int(11, 12),
        new Vector2Int(11, 13), new Vector2Int(13, 15),
        new Vector2Int(12, 14), new Vector2Int(14, 16),
        new Vector2Int(11, 23), new Vector2Int(12, 24),
        new Vector2Int(23, 24),
        new Vector2Int(23, 25), new Vector2Int(25, 27),
        new Vector2Int(24, 26), new Vector2Int(26, 28),
        new Vector2Int(0, 11), new Vector2Int(0, 12)
    };

    [Header("Camera")]
    [SerializeField] private TextAsset poseModel;
    [SerializeField] private RawImage preview;
    [SerializeField] private string preferredCameraName;
    [SerializeField] private bool mirrorPreview = true;
    [SerializeField, Min(320)] private int requestedWidth = 1280;
    [SerializeField, Min(240)] private int requestedHeight = 720;
    [SerializeField, Range(1, 60)] private int requestedFps = 30;
    [SerializeField, Min(0.03f)] private float detectionInterval = 0.12f;

    [Header("MediaPipe - CPU")]
    [SerializeField, Range(0f, 1f)]
    private float minPoseDetectionConfidence = 0.3f;

    [SerializeField, Range(0f, 1f)]
    private float minPosePresenceConfidence = 0.5f;

    [SerializeField, Range(0f, 1f)]
    private float minTrackingConfidence = 0.3f;

    [Header("Correct position")]
    [SerializeField, Range(0f, 1f)] private float minVisibility = 0.5f;
    [SerializeField, Range(0f, 0.5f)] private float horizontalCenterTolerance = 0.12f;
    [SerializeField, Range(0.2f, 1f)] private float minimumBodyHeight = 0.52f;
    [SerializeField, Range(0.2f, 1f)] private float maximumBodyHeight = 0.92f;
    [SerializeField, Range(0.001f, 0.2f)] private float maximumMovement = 0.025f;
    [SerializeField, Min(0.25f)] private float requiredHoldSeconds = 2f;

    [Header("Optional UI")]
    [SerializeField] private Text statusText;
    [SerializeField] private bool drawDebugOverlay = true;
    [SerializeField] private UnityEvent onPositionConfirmed;

    private readonly List<NormalizedLandmark> latestLandmarks =
        new List<NormalizedLandmark>(LandmarkCount);

    private readonly Vector2[] previousPositions =
        new Vector2[StabilityLandmarks.Length];

    private readonly Vector3[] previewCorners = new Vector3[4];

    private WebCamTexture webCamTexture;
    private PoseLandmarker poseLandmarker;
    private TextureFrame textureFrame;
    private Coroutine detectionCoroutine;
    private bool hasPreviousPose;
    private bool isInitializing;
    private float holdTimer;
    private string statusMessage = "Đang khởi động camera...";

    public IReadOnlyList<NormalizedLandmark> LatestLandmarks => latestLandmarks;
    public bool IsPositionConfirmed { get; private set; }
    public float HoldProgress => requiredHoldSeconds <= 0f
        ? 1f
        : Mathf.Clamp01(holdTimer / requiredHoldSeconds);

    private IEnumerator Start()
    {
        yield return InitializeAndRun();
    }

    [ContextMenu("Restart Positioning Step")]
    public void RestartPositioning()
    {
        StopDetection();
        IsPositionConfirmed = false;
        holdTimer = 0f;
        hasPreviousPose = false;
        latestLandmarks.Clear();
        detectionCoroutine = StartCoroutine(InitializeAndRun());
    }

    private IEnumerator InitializeAndRun()
    {
        if (isInitializing)
        {
            yield break;
        }

        isInitializing = true;
        SetStatus("Đang xin quyền truy cập camera...");

        if (poseModel == null || preview == null)
        {
            SetStatus("Thiếu Pose Model hoặc Preview trong Inspector.");
            Debug.LogError(statusMessage, this);
            isInitializing = false;
            yield break;
        }

        yield return RequestCameraPermission();

        if (!HasCameraPermission())
        {
            SetStatus("Ứng dụng chưa được cấp quyền camera.");
            Debug.LogError(statusMessage, this);
            isInitializing = false;
            yield break;
        }

        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices == null || devices.Length == 0)
        {
            SetStatus("Không tìm thấy camera.");
            Debug.LogError(statusMessage, this);
            isInitializing = false;
            yield break;
        }

        string cameraName = FindCameraName(devices);
        webCamTexture = new WebCamTexture(
            cameraName,
            requestedWidth,
            requestedHeight,
            requestedFps
        );
        webCamTexture.Play();

        float timeoutAt = Time.realtimeSinceStartup + 10f;
        while (webCamTexture.width <= 16 && Time.realtimeSinceStartup < timeoutAt)
        {
            yield return null;
        }

        if (webCamTexture.width <= 16)
        {
            SetStatus("Camera không trả về hình ảnh.");
            Debug.LogError(statusMessage, this);
            isInitializing = false;
            yield break;
        }

        preview.texture = webCamTexture;
        preview.uvRect = mirrorPreview
            ? new Rect(1f, 0f, -1f, 1f)
            : new Rect(0f, 0f, 1f, 1f);

        try
        {
            var baseOptions = new BaseOptions(
                delegateCase: BaseOptions.Delegate.CPU,
                modelAssetBuffer: poseModel.bytes
            );

            var options = new PoseLandmarkerOptions(
                baseOptions: baseOptions,
                runningMode: RunningMode.VIDEO,
                numPoses: 1,
                minPoseDetectionConfidence: minPoseDetectionConfidence,
                minPosePresenceConfidence: minPosePresenceConfidence,
                minTrackingConfidence: minTrackingConfidence,
                outputSegmentationMasks: false
            );

            poseLandmarker = PoseLandmarker.CreateFromOptions(options);
            textureFrame = new TextureFrame(
                webCamTexture.width,
                webCamTexture.height,
                TextureFormat.RGBA32
            );
        }
        catch (Exception exception)
        {
            SetStatus("Không thể khởi tạo nhận diện pose.");
            Debug.LogError(statusMessage, this);
            Debug.LogException(exception, this);
            isInitializing = false;
            yield break;
        }

        isInitializing = false;
        SetStatus("Vui lòng di chuyển vào giữa khung hình.");
        detectionCoroutine = StartCoroutine(DetectionLoop());
    }

    private IEnumerator DetectionLoop()
    {
        var waitForEndOfFrame = new WaitForEndOfFrame();
        float nextDetectionTime = 0f;

        while (enabled && webCamTexture != null && webCamTexture.isPlaying)
        {
            yield return waitForEndOfFrame;

            if (!webCamTexture.didUpdateThisFrame ||
                Time.unscaledTime < nextDetectionTime)
            {
                continue;
            }

            nextDetectionTime = Time.unscaledTime + detectionInterval;

            try
            {
                textureFrame.ReadTextureOnCPU(
                    webCamTexture,
                    flipHorizontally: mirrorPreview,
                    flipVertically: webCamTexture.videoVerticallyMirrored
                );

                using var mediaPipeImage = textureFrame.BuildCPUImage();
                long timestampMilliseconds =
                    (long)(Time.realtimeSinceStartupAsDouble * 1000d);

                PoseLandmarkerResult result = poseLandmarker.DetectForVideo(
                    mediaPipeImage,
                    timestampMilliseconds,
                    new ImageProcessingOptions(rotationDegrees: 0)
                );

                ProcessDetectionResult(result);
            }
            catch (Exception exception)
            {
                SetStatus("Nhận diện camera gặp lỗi.");
                Debug.LogError(statusMessage, this);
                Debug.LogException(exception, this);
                yield break;
            }
        }
    }

    private void ProcessDetectionResult(PoseLandmarkerResult result)
    {
        if (result.poseLandmarks == null || result.poseLandmarks.Count == 0)
        {
            latestLandmarks.Clear();
            ResetHold("Chưa thấy người. Hãy bước vào khung hình.");
            return;
        }

        List<NormalizedLandmark> detected = result.poseLandmarks[0].landmarks;
        if (detected == null || detected.Count < LandmarkCount)
        {
            latestLandmarks.Clear();
            ResetHold("Chưa nhận diện được toàn bộ cơ thể.");
            return;
        }

        latestLandmarks.Clear();
        latestLandmarks.AddRange(detected);

        if (IsPositionConfirmed)
        {
            return;
        }

        string correction = EvaluatePosition(detected);
        bool stable = IsPoseStable(detected);
        SavePreviousPositions(detected);

        if (!string.IsNullOrEmpty(correction))
        {
            ResetHold(correction);
            return;
        }

        if (!stable)
        {
            ResetHold("Đúng vị trí. Hãy đứng yên...");
            return;
        }

        holdTimer += detectionInterval;
        float remaining = Mathf.Max(0f, requiredHoldSeconds - holdTimer);
        SetStatus($"Giữ nguyên vị trí... {remaining:0.0}s");

        if (holdTimer < requiredHoldSeconds)
        {
            return;
        }

        IsPositionConfirmed = true;
        holdTimer = requiredHoldSeconds;
        SetStatus("Đã vào đúng vị trí!");
        Debug.Log("BƯỚC 1 HOÀN TẤT: Người chơi đã vào đúng vị trí.", this);
        onPositionConfirmed?.Invoke();
    }

    private string EvaluatePosition(IReadOnlyList<NormalizedLandmark> landmarks)
    {
        foreach (int index in RequiredLandmarks)
        {
            if ((landmarks[index].visibility ?? 0f) < minVisibility)
            {
                return "Hãy lùi lại để camera thấy rõ toàn thân.";
            }
        }

        float shoulderCenterX = (landmarks[11].x + landmarks[12].x) * 0.5f;
        float hipCenterX = (landmarks[23].x + landmarks[24].x) * 0.5f;
        float bodyCenterX = (shoulderCenterX + hipCenterX) * 0.5f;

        if (Mathf.Abs(bodyCenterX - 0.5f) > horizontalCenterTolerance)
        {
            return "Hãy dịch người vào giữa khung hình.";
        }

        float ankleY = Mathf.Max(landmarks[27].y, landmarks[28].y);
        float bodyHeight = ankleY - landmarks[0].y;

        if (bodyHeight < minimumBodyHeight)
        {
            return "Bạn đang quá xa. Hãy tiến gần camera hơn.";
        }

        if (bodyHeight > maximumBodyHeight ||
            landmarks[0].y < 0.02f || ankleY > 0.98f)
        {
            return "Bạn đang quá gần. Hãy lùi ra xa camera.";
        }

        return string.Empty;
    }

    private bool IsPoseStable(IReadOnlyList<NormalizedLandmark> landmarks)
    {
        if (!hasPreviousPose)
        {
            return false;
        }

        float totalMovement = 0f;
        foreach (int index in StabilityLandmarks)
        {
            int previousIndex = Array.IndexOf(StabilityLandmarks, index);
            Vector2 current = new Vector2(landmarks[index].x, landmarks[index].y);
            totalMovement += Vector2.Distance(current, previousPositions[previousIndex]);
        }

        float averageMovement = totalMovement / StabilityLandmarks.Length;
        return averageMovement <= maximumMovement;
    }

    private void SavePreviousPositions(IReadOnlyList<NormalizedLandmark> landmarks)
    {
        for (int i = 0; i < StabilityLandmarks.Length; i++)
        {
            NormalizedLandmark landmark = landmarks[StabilityLandmarks[i]];
            previousPositions[i] = new Vector2(landmark.x, landmark.y);
        }

        hasPreviousPose = true;
    }

    private void ResetHold(string message)
    {
        holdTimer = 0f;
        SetStatus(message);
    }

    private void SetStatus(string message)
    {
        statusMessage = message;
        if (statusText != null)
        {
            statusText.text = message;
        }
    }

    private string FindCameraName(IReadOnlyList<WebCamDevice> devices)
    {
        if (!string.IsNullOrWhiteSpace(preferredCameraName))
        {
            for (int i = 0; i < devices.Count; i++)
            {
                if (devices[i].name.IndexOf(
                    preferredCameraName,
                    StringComparison.OrdinalIgnoreCase
                ) >= 0)
                {
                    return devices[i].name;
                }
            }
        }

        return devices[0].name;
    }

    private static IEnumerator RequestCameraPermission()
    {
#if UNITY_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
            yield return new WaitForSeconds(0.2f);
        }
#else
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        }
#endif
    }

    private static bool HasCameraPermission()
    {
#if UNITY_ANDROID
        return Permission.HasUserAuthorizedPermission(Permission.Camera);
#else
        return Application.HasUserAuthorization(UserAuthorization.WebCam);
#endif
    }

    private Rect GetPreviewScreenRect()
    {
        preview.rectTransform.GetWorldCorners(previewCorners);
        Camera canvasCamera = preview.canvas.renderMode == RenderMode.ScreenSpaceOverlay
            ? null
            : preview.canvas.worldCamera;

        Vector2 bottomLeft = RectTransformUtility.WorldToScreenPoint(
            canvasCamera,
            previewCorners[0]
        );
        Vector2 topRight = RectTransformUtility.WorldToScreenPoint(
            canvasCamera,
            previewCorners[2]
        );

        return new Rect(
            bottomLeft.x,
            Screen.height - topRight.y,
            topRight.x - bottomLeft.x,
            topRight.y - bottomLeft.y
        );
    }

    private void OnGUI()
    {
        if (!drawDebugOverlay || preview == null)
        {
            return;
        }

        Rect feedRect = GetPreviewScreenRect();
        if (feedRect.width <= 0f || feedRect.height <= 0f)
        {
            return;
        }

        Color guideColor = IsPositionConfirmed
            ? new Color(0.2f, 1f, 0.35f, 0.95f)
            : new Color(1f, 0.82f, 0.15f, 0.95f);

        Rect guideRect = new Rect(
            feedRect.x + feedRect.width * (0.5f - horizontalCenterTolerance),
            feedRect.y + feedRect.height * 0.04f,
            feedRect.width * horizontalCenterTolerance * 2f,
            feedRect.height * 0.92f
        );
        DrawRectBorder(guideRect, 3f, guideColor);

        if (latestLandmarks.Count >= LandmarkCount)
        {
            DrawSkeleton(feedRect);
        }

        float panelWidth = Mathf.Min(feedRect.width * 0.9f, 720f);
        var statusRect = new Rect(
            feedRect.center.x - panelWidth * 0.5f,
            feedRect.yMax - 92f,
            panelWidth,
            54f
        );

        var style = new GUIStyle(GUI.skin.box)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = Mathf.Clamp(Screen.height / 35, 18, 34),
            fontStyle = FontStyle.Bold,
            wordWrap = true
        };
        GUI.Box(statusRect, statusMessage, style);

        if (!IsPositionConfirmed && holdTimer > 0f)
        {
            Rect progressBackground = new Rect(
                statusRect.x,
                statusRect.yMax + 6f,
                statusRect.width,
                10f
            );
            GUI.color = new Color(0f, 0f, 0f, 0.65f);
            GUI.DrawTexture(progressBackground, Texture2D.whiteTexture);
            GUI.color = guideColor;
            GUI.DrawTexture(
                new Rect(
                    progressBackground.x,
                    progressBackground.y,
                    progressBackground.width * HoldProgress,
                    progressBackground.height
                ),
                Texture2D.whiteTexture
            );
            GUI.color = Color.white;
        }
    }

    private void DrawSkeleton(Rect feedRect)
    {
        Color skeletonColor = new Color(0.15f, 1f, 0.35f, 0.9f);

        foreach (Vector2Int connection in SkeletonConnections)
        {
            NormalizedLandmark start = latestLandmarks[connection.x];
            NormalizedLandmark end = latestLandmarks[connection.y];
            if ((start.visibility ?? 0f) < 0.3f ||
                (end.visibility ?? 0f) < 0.3f)
            {
                continue;
            }

            DrawGuiLine(
                LandmarkToScreen(start, feedRect),
                LandmarkToScreen(end, feedRect),
                skeletonColor,
                4f
            );
        }

        foreach (int index in StabilityLandmarks)
        {
            NormalizedLandmark landmark = latestLandmarks[index];
            if ((landmark.visibility ?? 0f) < 0.3f)
            {
                continue;
            }

            Vector2 point = LandmarkToScreen(landmark, feedRect);
            GUI.color = skeletonColor;
            GUI.DrawTexture(
                new Rect(point.x - 5f, point.y - 5f, 10f, 10f),
                Texture2D.whiteTexture
            );
            GUI.color = Color.white;
        }
    }

    private static Vector2 LandmarkToScreen(
        NormalizedLandmark landmark,
        Rect feedRect)
    {
        return new Vector2(
            feedRect.x + landmark.x * feedRect.width,
            feedRect.y + landmark.y * feedRect.height
        );
    }

    private static void DrawGuiLine(
        Vector2 start,
        Vector2 end,
        Color color,
        float thickness)
    {
        Matrix4x4 previousMatrix = GUI.matrix;
        Color previousColor = GUI.color;
        float angle = Mathf.Atan2(end.y - start.y, end.x - start.x) * Mathf.Rad2Deg;
        float length = Vector2.Distance(start, end);

        GUI.color = color;
        GUIUtility.RotateAroundPivot(angle, start);
        GUI.DrawTexture(
            new Rect(start.x, start.y - thickness * 0.5f, length, thickness),
            Texture2D.whiteTexture
        );
        GUI.matrix = previousMatrix;
        GUI.color = previousColor;
    }

    private static void DrawRectBorder(Rect rect, float thickness, Color color)
    {
        Color previousColor = GUI.color;
        GUI.color = color;
        GUI.DrawTexture(new Rect(rect.x, rect.y, rect.width, thickness), Texture2D.whiteTexture);
        GUI.DrawTexture(new Rect(rect.x, rect.yMax - thickness, rect.width, thickness), Texture2D.whiteTexture);
        GUI.DrawTexture(new Rect(rect.x, rect.y, thickness, rect.height), Texture2D.whiteTexture);
        GUI.DrawTexture(new Rect(rect.xMax - thickness, rect.y, thickness, rect.height), Texture2D.whiteTexture);
        GUI.color = previousColor;
    }

    private void StopDetection()
    {
        if (detectionCoroutine != null)
        {
            StopCoroutine(detectionCoroutine);
            detectionCoroutine = null;
        }

        poseLandmarker?.Close();
        poseLandmarker = null;
        textureFrame?.Dispose();
        textureFrame = null;

        if (webCamTexture != null)
        {
            webCamTexture.Stop();
            Destroy(webCamTexture);
            webCamTexture = null;
        }
    }

    private void OnDestroy()
    {
        StopDetection();
    }
}
