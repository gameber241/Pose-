using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public sealed class LobbyController : MonoBehaviour
{
    [SerializeField] private Button playGameButton;
    [SerializeField] private Button genPoseButton;

    private void Awake()
    {
        if (playGameButton != null)
        {
            playGameButton.onClick.AddListener(OpenGame);
        }

        if (genPoseButton != null)
        {
            genPoseButton.onClick.AddListener(OpenGenPose);
        }
    }

    private void OnDestroy()
    {
        if (playGameButton != null)
        {
            playGameButton.onClick.RemoveListener(OpenGame);
        }

        if (genPoseButton != null)
        {
            genPoseButton.onClick.RemoveListener(OpenGenPose);
        }
    }

    public void OpenGame()
    {
        SceneManager.LoadScene("Game");
    }

    public void OpenGenPose()
    {
        SceneManager.LoadScene("CreateDataPose");
    }
}
