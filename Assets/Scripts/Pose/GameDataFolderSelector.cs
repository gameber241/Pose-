using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

[DefaultExecutionOrder(-1000)]
public sealed class GameDataFolderSelector : MonoBehaviour
{
    private sealed class LoadedPoseFolder
    {
        public readonly Dictionary<string, TextAsset> ByName =
            new Dictionary<string, TextAsset>(StringComparer.OrdinalIgnoreCase);
        public readonly List<TextAsset> InOrder = new List<TextAsset>();
        public int FileCount;
    }

    [SerializeField] private PosePositioningStep positioningStep;
    [SerializeField] private bool chooseFolderOnStart;

    private readonly List<UnityEngine.Object> runtimeAssets =
        new List<UnityEngine.Object>();

    private string status = "Chọn folder chứa các file JSON pose.";
    private bool showSelection = true;
    private Vector2 uiScrollPosition;

    private void Awake()
    {
        if (positioningStep == null)
        {
            positioningStep = GetComponent<PosePositioningStep>();
        }

        if (positioningStep != null)
        {
            positioningStep.enabled = false;
        }
    }

    private void Start()
    {
        if (chooseFolderOnStart)
        {
            ChooseDataFolder();
        }
    }

    public void ChooseDataFolder()
    {
        string folder = NativeFolderDialog.Open("Chọn folder data để chơi");
        if (string.IsNullOrWhiteSpace(folder))
        {
            status = "Chưa chọn folder data.";
            showSelection = true;
            return;
        }

        try
        {
            if (positioningStep == null)
            {
                status = "Scene Game thiếu PosePositioningStep.";
                showSelection = true;
                return;
            }

            LoadedPoseFolder referencePoses = LoadReferencePoses(folder);
            int configuredCount = positioningStep.ConfigureReferencePoses(referencePoses.ByName);
            string loadMode = "trung ten";
            if (configuredCount == 0)
            {
                configuredCount =
                    positioningStep.ConfigureReferencePosesInOrder(referencePoses.InOrder);
                loadMode = "theo thu tu";
            }
            if (configuredCount == 0)
            {
                status =
                    $"Không nạp được data. Folder có {referencePoses.FileCount} JSON, " +
                    $"hợp lệ {referencePoses.InOrder.Count}.";
                showSelection = true;
                return;
            }

            positioningStep.enabled = true;
            showSelection = false;
            status = $"Đã nạp {configuredCount} JSON pose ({loadMode}) từ {folder}.";
            Debug.Log(status, this);
        }
        catch (Exception exception)
        {
            status = $"Không thể đọc folder data: {exception.Message}";
            showSelection = true;
            Debug.LogException(exception, this);
        }
    }

    private LoadedPoseFolder LoadReferencePoses(string folder)
    {
        ReleaseRuntimeAssets();

        string[] jsonFiles = Directory.GetFiles(folder, "*.json", SearchOption.AllDirectories);
        Array.Sort(jsonFiles, ComparePosePaths);
        var result = new LoadedPoseFolder
        {
            FileCount = jsonFiles.Length
        };

        foreach (string jsonFile in jsonFiles)
        {
            string poseName = Path.GetFileNameWithoutExtension(jsonFile);
            string json = File.ReadAllText(jsonFile);
            SavedPoseData data = JsonUtility.FromJson<SavedPoseData>(json);
            if (data == null || data.landmarks == null || data.landmarks.Count < 33)
            {
                Debug.LogWarning($"Bỏ qua JSON pose không hợp lệ: {jsonFile}", this);
                continue;
            }

            var poseJson = new TextAsset(json) { name = poseName };
            runtimeAssets.Add(poseJson);
            result.ByName[poseName] = poseJson;
            result.InOrder.Add(poseJson);
        }

        return result;
    }

    private static int ComparePosePaths(string left, string right)
    {
        string leftName = Path.GetFileNameWithoutExtension(left);
        string rightName = Path.GetFileNameWithoutExtension(right);
        if (int.TryParse(leftName, out int leftNumber) &&
            int.TryParse(rightName, out int rightNumber))
        {
            return leftNumber.CompareTo(rightNumber);
        }

        return string.Compare(leftName, rightName, StringComparison.OrdinalIgnoreCase);
    }

    private void OnGUI()
    {
        if (!showSelection)
        {
            return;
        }

        const float uiScale = 1.5f;
        float width = Mathf.Min(620f * uiScale, Mathf.Max(320f, Screen.width - 40f));
        float height = Mathf.Min(300f * uiScale, Mathf.Max(260f, Screen.height - 40f));
        var area = new Rect(
            (Screen.width - width) * 0.5f,
            (Screen.height - height) * 0.5f,
            width,
            height);

        GUILayout.BeginArea(area, GUI.skin.box);
        uiScrollPosition = GUILayout.BeginScrollView(uiScrollPosition);
        GUILayout.Space(12f * uiScale);
        GUILayout.Label("CHỌN DATA POSE", CreateTitleStyle(uiScale),
            GUILayout.Height(34f * uiScale));
        GUILayout.Space(10f * uiScale);
        GUILayout.Label(status, CreateMessageStyle(uiScale),
            GUILayout.MinHeight(64f * uiScale));
        GUILayout.FlexibleSpace();
        if (GUILayout.Button("Chọn Folder Data", GUILayout.Height(50f * uiScale)))
        {
            ChooseDataFolder();
        }

        if (GUILayout.Button("Về Lobby", GUILayout.Height(44f * uiScale)))
        {
            SceneManager.LoadScene("Lobby");
        }
        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }

    private static GUIStyle CreateTitleStyle(float scale)
    {
        return new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = Mathf.RoundToInt(24f * scale),
            fontStyle = FontStyle.Bold
        };
    }

    private static GUIStyle CreateMessageStyle(float scale)
    {
        return new GUIStyle(GUI.skin.label)
        {
            alignment = TextAnchor.MiddleCenter,
            fontSize = Mathf.RoundToInt(16f * scale),
            wordWrap = true
        };
    }

    private void ReleaseRuntimeAssets()
    {
        foreach (UnityEngine.Object asset in runtimeAssets)
        {
            if (asset != null)
            {
                Destroy(asset);
            }
        }

        runtimeAssets.Clear();
    }

    private void OnDestroy()
    {
        ReleaseRuntimeAssets();
    }
}
