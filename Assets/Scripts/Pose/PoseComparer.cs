using System;
using System.Collections.Generic;
using Mediapipe.Tasks.Components.Containers;
using UnityEngine;

/// <summary>
/// Result shape equivalent to computePoseSimilarity() in lib/poseSimilarity.js.
/// </summary>
[Serializable]
public sealed class PoseCompareResult
{
    public int score;
    public int angleScore;
    public int boneDirectionScore;
    public int positionScore;
    public int mirrorBonusScore;
    public bool isMirrored;

    // The web code does not expose these, but they are useful when debugging Unity.
    public float rawScore;
    public float curvedScore;
    public string grade;
}

/// <summary>
/// Exact C# port of lib/poseSimilarity.js.
///
/// Important: the web formula intentionally totals 90% before the optional
/// mirror bonus (40% + 35% + 15%). Consequently, an identical non-mirrored
/// pose scores 85 after the 1.5 power curve. This class preserves that behavior
/// so Unity and the current web project return the same score.
/// </summary>
public static class PoseComparer
{
    private const int LandmarkCount = 33;
    private const float MinimumVisibility = 0.1f;
    private const float BoundingBoxVisibility = 0.3f;
    private const int MidHip = -1;
    private const int MidShoulder = -2;

    private readonly struct JointDefinition
    {
        public readonly int pointA;
        public readonly int joint;
        public readonly int pointB;

        public JointDefinition(int pointA, int joint, int pointB)
        {
            this.pointA = pointA;
            this.joint = joint;
            this.pointB = pointB;
        }
    }

    private readonly struct BoneDefinition
    {
        public readonly int start;
        public readonly int end;

        public BoneDefinition(int start, int end)
        {
            this.start = start;
            this.end = end;
        }
    }

    private struct Point
    {
        public float x;
        public float y;
        public float z;
        public float visibility;
        public float presence;
    }

    private struct NormalizedPoint
    {
        public float nx;
        public float ny;
        public float visibility;
    }

    private readonly struct MirrorResult
    {
        public readonly bool isMirrored;
        public readonly float bonus;

        public MirrorResult(bool isMirrored, float bonus)
        {
            this.isMirrored = isMirrored;
            this.bonus = bonus;
        }
    }

    // Same order and landmark definitions as JOINT_ANGLES in poseSimilarity.js.
    private static readonly JointDefinition[] Joints =
    {
        new JointDefinition(23, 11, 13),
        new JointDefinition(24, 12, 14),
        new JointDefinition(11, 13, 15),
        new JointDefinition(12, 14, 16),
        new JointDefinition(11, 23, 25),
        new JointDefinition(12, 24, 26),
        new JointDefinition(23, 25, 27),
        new JointDefinition(24, 26, 28),
        new JointDefinition(23, MidHip, MidShoulder),
        new JointDefinition(11, MidShoulder, 0)
    };

    // Same order and landmark definitions as BONE_SEGMENTS in poseSimilarity.js.
    private static readonly BoneDefinition[] Bones =
    {
        new BoneDefinition(11, 13),
        new BoneDefinition(13, 15),
        new BoneDefinition(12, 14),
        new BoneDefinition(14, 16),
        new BoneDefinition(11, 12),
        new BoneDefinition(23, 24),
        new BoneDefinition(11, 23),
        new BoneDefinition(12, 24),
        new BoneDefinition(23, 25),
        new BoneDefinition(25, 27),
        new BoneDefinition(24, 26),
        new BoneDefinition(26, 28),
        new BoneDefinition(0, 11),
        new BoneDefinition(0, 12)
    };

    private static readonly int[] MajorPositionIndices =
    {
        0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
    };

    public static PoseCompareResult Compare(
        TextAsset referencePoseJson,
        IReadOnlyList<NormalizedLandmark> userLandmarks)
    {
        if (referencePoseJson == null)
        {
            Debug.LogError("Reference Pose JSON đang bị null.");
            return CreateEmptyResult();
        }

        if (userLandmarks == null || userLandmarks.Count < LandmarkCount)
        {
            Debug.LogError(
                $"Pose người chơi không đủ landmark: " +
                $"{userLandmarks?.Count ?? 0}/33."
            );
            return CreateEmptyResult();
        }

        SavedPoseData savedPose;

        try
        {
            savedPose = JsonUtility.FromJson<SavedPoseData>(referencePoseJson.text);
        }
        catch (Exception exception)
        {
            Debug.LogError("Không đọc được dữ liệu pose mẫu.");
            Debug.LogException(exception);
            return CreateEmptyResult();
        }

        if (savedPose == null ||
            savedPose.landmarks == null ||
            savedPose.landmarks.Count < LandmarkCount)
        {
            Debug.LogError(
                "Pose mẫu phải là một JSON object có trường landmarks chứa đủ 33 điểm."
            );
            return CreateEmptyResult();
        }

        Point[] reference = ConvertReferencePose(savedPose);
        Point[] user = ConvertUserPose(userLandmarks);
        return ComputePoseSimilarity(reference, user);
    }

    private static PoseCompareResult ComputePoseSimilarity(
        Point[] reference,
        Point[] user)
    {
        MirrorResult mirror = ComputeMirrorBonus(reference, user);
        Point[] effectiveUser = mirror.isMirrored
            ? MirrorLandmarks(user)
            : user;

        float angleScore = ScoreJointAngles(reference, effectiveUser);
        float boneScore = ScoreBoneDirections(reference, effectiveUser);
        float positionScore = ScoreRelativePositions(reference, effectiveUser);

        // Intentionally identical to the web formula. Base weights total 0.90.
        float rawScore =
            angleScore * 0.40f +
            boneScore * 0.35f +
            positionScore * 0.15f +
            mirror.bonus * 0.10f;

        float curvedScore = Mathf.Pow(Mathf.Clamp01(rawScore), 1.5f);
        int finalScore = WebRound(Mathf.Min(curvedScore * 100f, 100f));

        return new PoseCompareResult
        {
            score = finalScore,
            angleScore = WebRound(angleScore * 100f),
            boneDirectionScore = WebRound(boneScore * 100f),
            positionScore = WebRound(positionScore * 100f),
            mirrorBonusScore = WebRound(mirror.bonus * 100f),
            isMirrored = mirror.isMirrored,
            rawScore = rawScore,
            curvedScore = curvedScore,
            grade = GetScoreGrade(finalScore)
        };
    }

    private static float ScoreJointAngles(Point[] reference, Point[] user)
    {
        float totalScore = 0f;
        float totalWeight = 0f;

        foreach (JointDefinition definition in Joints)
        {
            Point referenceA = GetPoint(reference, definition.pointA);
            Point referenceJoint = GetPoint(reference, definition.joint);
            Point referenceB = GetPoint(reference, definition.pointB);
            Point userA = GetPoint(user, definition.pointA);
            Point userJoint = GetPoint(user, definition.joint);
            Point userB = GetPoint(user, definition.pointB);

            float weight = Minimum(
                referenceA.visibility,
                referenceJoint.visibility,
                referenceB.visibility,
                userA.visibility,
                userJoint.visibility,
                userB.visibility
            );

            if (weight < MinimumVisibility)
            {
                continue;
            }

            float referenceAngle = CalculateAngle(
                referenceA,
                referenceJoint,
                referenceB
            );
            float userAngle = CalculateAngle(userA, userJoint, userB);
            float difference = Mathf.Abs(referenceAngle - userAngle) / Mathf.PI;
            float score = 1f - difference;

            totalScore += score * weight;
            totalWeight += weight;
        }

        return totalWeight > 0f ? totalScore / totalWeight : 0.5f;
    }

    private static float ScoreBoneDirections(Point[] reference, Point[] user)
    {
        float totalScore = 0f;
        float totalWeight = 0f;

        foreach (BoneDefinition definition in Bones)
        {
            Point referenceStart = reference[definition.start];
            Point referenceEnd = reference[definition.end];
            Point userStart = user[definition.start];
            Point userEnd = user[definition.end];

            float weight = Minimum(
                referenceStart.visibility,
                referenceEnd.visibility,
                userStart.visibility,
                userEnd.visibility
            );

            if (weight < MinimumVisibility)
            {
                continue;
            }

            Vector2 referenceDirection = NormalizeLikeWeb(
                new Vector2(
                    referenceEnd.x - referenceStart.x,
                    referenceEnd.y - referenceStart.y
                )
            );
            Vector2 userDirection = NormalizeLikeWeb(
                new Vector2(
                    userEnd.x - userStart.x,
                    userEnd.y - userStart.y
                )
            );

            float score = (Vector2.Dot(referenceDirection, userDirection) + 1f) / 2f;
            totalScore += score * weight;
            totalWeight += weight;
        }

        return totalWeight > 0f ? totalScore / totalWeight : 0.5f;
    }

    private static float ScoreRelativePositions(Point[] reference, Point[] user)
    {
        NormalizedPoint[] normalizedReference = NormalizeToBoundingBox(reference);
        NormalizedPoint[] normalizedUser = NormalizeToBoundingBox(user);
        float totalScore = 0f;
        float totalWeight = 0f;

        foreach (int index in MajorPositionIndices)
        {
            NormalizedPoint referencePoint = normalizedReference[index];
            NormalizedPoint userPoint = normalizedUser[index];
            float weight = Mathf.Min(
                referencePoint.visibility,
                userPoint.visibility
            );

            if (weight < MinimumVisibility)
            {
                continue;
            }

            float deltaX = referencePoint.nx - userPoint.nx;
            float deltaY = referencePoint.ny - userPoint.ny;
            float distance = Mathf.Sqrt(deltaX * deltaX + deltaY * deltaY);
            float score = Mathf.Max(0f, 1f - distance / 0.5f);

            totalScore += score * weight;
            totalWeight += weight;
        }

        return totalWeight > 0f ? totalScore / totalWeight : 0.5f;
    }

    private static NormalizedPoint[] NormalizeToBoundingBox(Point[] points)
    {
        int visibleCount = 0;
        float minX = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float minY = float.PositiveInfinity;
        float maxY = float.NegativeInfinity;

        foreach (Point point in points)
        {
            if (point.visibility <= BoundingBoxVisibility)
            {
                continue;
            }

            visibleCount++;
            minX = Mathf.Min(minX, point.x);
            maxX = Mathf.Max(maxX, point.x);
            minY = Mathf.Min(minY, point.y);
            maxY = Mathf.Max(maxY, point.y);
        }

        var result = new NormalizedPoint[points.Length];

        // Well-formed MediaPipe poses always have at least two visible points.
        // This safe fallback preserves the original coordinates for malformed input.
        if (visibleCount < 2)
        {
            for (int i = 0; i < points.Length; i++)
            {
                result[i] = new NormalizedPoint
                {
                    nx = points[i].x,
                    ny = points[i].y,
                    visibility = points[i].visibility
                };
            }
            return result;
        }

        float rangeX = maxX - minX;
        float rangeY = maxY - minY;

        if (rangeX == 0f)
        {
            rangeX = 1f;
        }
        if (rangeY == 0f)
        {
            rangeY = 1f;
        }

        for (int i = 0; i < points.Length; i++)
        {
            result[i] = new NormalizedPoint
            {
                nx = (points[i].x - minX) / rangeX,
                ny = (points[i].y - minY) / rangeY,
                visibility = points[i].visibility
            };
        }

        return result;
    }

    private static MirrorResult ComputeMirrorBonus(Point[] reference, Point[] user)
    {
        Point[] mirroredUser = MirrorLandmarks(user);
        float normalAngleScore = ScoreJointAngles(reference, user);
        float mirroredAngleScore = ScoreJointAngles(reference, mirroredUser);

        if (mirroredAngleScore > normalAngleScore + 0.1f)
        {
            return new MirrorResult(
                true,
                (mirroredAngleScore - normalAngleScore) * 0.5f
            );
        }

        return new MirrorResult(false, 0f);
    }

    private static Point[] MirrorLandmarks(Point[] source)
    {
        var mirrored = new Point[source.Length];

        for (int index = 0; index < source.Length; index++)
        {
            int pairedIndex = GetWebMirrorIndex(index);
            Point sourcePoint = pairedIndex >= 0 ? source[pairedIndex] : source[index];

            mirrored[index] = new Point
            {
                x = 1f - sourcePoint.x,
                y = sourcePoint.y,
                z = sourcePoint.z,
                visibility = sourcePoint.visibility,
                presence = sourcePoint.presence
            };
        }

        return mirrored;
    }

    // The web MIRROR_MAP intentionally swaps only body landmarks 11..32.
    private static int GetWebMirrorIndex(int index)
    {
        switch (index)
        {
            case 11: return 12;
            case 12: return 11;
            case 13: return 14;
            case 14: return 13;
            case 15: return 16;
            case 16: return 15;
            case 17: return 18;
            case 18: return 17;
            case 19: return 20;
            case 20: return 19;
            case 21: return 22;
            case 22: return 21;
            case 23: return 24;
            case 24: return 23;
            case 25: return 26;
            case 26: return 25;
            case 27: return 28;
            case 28: return 27;
            case 29: return 30;
            case 30: return 29;
            case 31: return 32;
            case 32: return 31;
            default: return -1;
        }
    }

    private static Point GetPoint(Point[] points, int index)
    {
        if (index >= 0)
        {
            return points[index];
        }

        if (index == MidHip)
        {
            return Midpoint(points[23], points[24]);
        }

        if (index == MidShoulder)
        {
            return Midpoint(points[11], points[12]);
        }

        return new Point { x = 0.5f, y = 0.5f, visibility = 0f };
    }

    private static Point Midpoint(Point left, Point right)
    {
        return new Point
        {
            x = (left.x + right.x) / 2f,
            y = (left.y + right.y) / 2f,
            z = (left.z + right.z) / 2f,
            visibility = Mathf.Min(left.visibility, right.visibility),
            presence = Mathf.Min(left.presence, right.presence)
        };
    }

    private static float CalculateAngle(Point pointA, Point joint, Point pointB)
    {
        float firstX = pointA.x - joint.x;
        float firstY = pointA.y - joint.y;
        float secondX = pointB.x - joint.x;
        float secondY = pointB.y - joint.y;
        float dot = firstX * secondX + firstY * secondY;
        float firstMagnitude = Mathf.Sqrt(firstX * firstX + firstY * firstY) + 1e-8f;
        float secondMagnitude = Mathf.Sqrt(secondX * secondX + secondY * secondY) + 1e-8f;
        float cosine = Mathf.Clamp(dot / (firstMagnitude * secondMagnitude), -1f, 1f);
        return Mathf.Acos(cosine);
    }

    private static Vector2 NormalizeLikeWeb(Vector2 vector)
    {
        float magnitude = Mathf.Sqrt(vector.x * vector.x + vector.y * vector.y) + 1e-8f;
        return new Vector2(vector.x / magnitude, vector.y / magnitude);
    }

    private static Point[] ConvertReferencePose(SavedPoseData pose)
    {
        var result = new Point[LandmarkCount];

        for (int i = 0; i < LandmarkCount; i++)
        {
            PosePointData source = pose.landmarks[i];
            result[i] = new Point
            {
                x = source.x,
                y = source.y,
                z = source.z,
                visibility = source.visibility,
                presence = source.presence
            };
        }

        return result;
    }

    private static Point[] ConvertUserPose(IReadOnlyList<NormalizedLandmark> landmarks)
    {
        var result = new Point[LandmarkCount];

        for (int i = 0; i < LandmarkCount; i++)
        {
            NormalizedLandmark source = landmarks[i];
            result[i] = new Point
            {
                x = source.x,
                y = source.y,
                z = source.z,
                visibility = source.visibility ?? 0f,
                presence = source.presence ?? 0f
            };
        }

        return result;
    }

    private static float Minimum(params float[] values)
    {
        float minimum = float.MaxValue;
        foreach (float value in values)
        {
            minimum = Mathf.Min(minimum, value);
        }
        return minimum;
    }

    // JavaScript Math.round() behavior for the non-negative values used here.
    private static int WebRound(float value)
    {
        return Mathf.FloorToInt(value + 0.5f);
    }

    public static string GetScoreGrade(int score)
    {
        if (score >= 95) return "PERFECT!";
        if (score >= 80) return "GREAT!";
        if (score >= 60) return "GOOD";
        if (score >= 40) return "OKAY";
        return "TRY AGAIN";
    }

    private static PoseCompareResult CreateEmptyResult()
    {
        return new PoseCompareResult
        {
            score = 0,
            angleScore = 0,
            boneDirectionScore = 0,
            positionScore = 0,
            mirrorBonusScore = 0,
            isMirrored = false,
            rawScore = 0f,
            curvedScore = 0f,
            grade = GetScoreGrade(0)
        };
    }
}
