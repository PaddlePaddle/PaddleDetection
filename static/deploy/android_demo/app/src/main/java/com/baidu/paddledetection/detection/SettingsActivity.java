package com.baidu.paddledetection.detection;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.EditTextPreference;
import android.preference.ListPreference;
import android.preference.PreferenceManager;
import androidx.appcompat.app.ActionBar;

import android.widget.Toast;

import com.baidu.paddledetection.common.AppCompatPreferenceActivity;
import com.baidu.paddledetection.common.Utils;
import com.baidu.paddledetection.detection.R;

import java.util.ArrayList;
import java.util.List;

public class SettingsActivity extends AppCompatPreferenceActivity implements SharedPreferences.OnSharedPreferenceChangeListener {
    private static final String TAG = SettingsActivity.class.getSimpleName();

    static public int selectedModelIdx = -1;
    static public String modelDir = "";
    static public String labelPath = "";
    static public int cpuThreadNum = 0;
    static public String cpuPowerMode = "";
    static public int inputWidth = 0;
    static public int inputHeight = 0;
    static public float[] inputMean = new float[]{};
    static public float[] inputStd = new float[]{};
    static public float scoreThreshold = 0.0f;

    ListPreference lpChoosePreInstalledModel = null;
    EditTextPreference etModelDir = null;
    EditTextPreference etLabelPath = null;
    ListPreference lpCPUThreadNum = null;
    ListPreference lpCPUPowerMode = null;
    EditTextPreference etInputWidth = null;
    EditTextPreference etInputHeight = null;
    EditTextPreference etInputMean = null;
    EditTextPreference etInputStd = null;
    EditTextPreference etScoreThreshold = null;

    List<String> preInstalledModelDirs = null;
    List<String> preInstalledLabelPaths = null;
    List<String> preInstalledCPUThreadNums = null;
    List<String> preInstalledCPUPowerModes = null;
    List<String> preInstalledInputWidths = null;
    List<String> preInstalledInputHeights = null;
    List<String> preInstalledInputMeans = null;
    List<String> preInstalledInputStds = null;
    List<String> preInstalledScoreThresholds = null;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.settings);
        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

        // Initialize pre-installed models
        preInstalledModelDirs = new ArrayList<String>();
        preInstalledLabelPaths = new ArrayList<String>();
        preInstalledCPUThreadNums = new ArrayList<String>();
        preInstalledCPUPowerModes = new ArrayList<String>();
        preInstalledInputWidths = new ArrayList<String>();
        preInstalledInputHeights = new ArrayList<String>();
        preInstalledInputMeans = new ArrayList<String>();
        preInstalledInputStds = new ArrayList<String>();
        preInstalledScoreThresholds = new ArrayList<String>();
        preInstalledModelDirs.add(getString(R.string.MODEL_DIR_DEFAULT));
        preInstalledLabelPaths.add(getString(R.string.LABEL_PATH_DEFAULT));
        preInstalledCPUThreadNums.add(getString(R.string.CPU_THREAD_NUM_DEFAULT));
        preInstalledCPUPowerModes.add(getString(R.string.CPU_POWER_MODE_DEFAULT));
        preInstalledInputWidths.add(getString(R.string.INPUT_WIDTH_DEFAULT));
        preInstalledInputHeights.add(getString(R.string.INPUT_HEIGHT_DEFAULT));
        preInstalledInputMeans.add(getString(R.string.INPUT_MEAN_DEFAULT));
        preInstalledInputStds.add(getString(R.string.INPUT_STD_DEFAULT));
        preInstalledScoreThresholds.add(getString(R.string.SCORE_THRESHOLD_DEFAULT));
        // Add yolov3_mobilenet_v3_for_hybrid_cpu_npu for CPU and huawei NPU
        if (Utils.isSupportedNPU()) {
            preInstalledModelDirs.add("models/yolov3_mobilenet_v3_for_hybrid_cpu_npu");
            preInstalledLabelPaths.add("labels/coco-labels-2014_2017.txt");
            preInstalledCPUThreadNums.add("1"); // Useless for NPU
            preInstalledCPUPowerModes.add("LITE_POWER_HIGH");  // Useless for NPU
            preInstalledInputWidths.add("320");
            preInstalledInputHeights.add("320");
            preInstalledInputMeans.add("0.485,0.456,0.406");
            preInstalledInputStds.add("0.229,0.224,0.225");
            preInstalledScoreThresholds.add("0.2");
        } else {
            Toast.makeText(this, "NPU model is not supported by your device.", Toast.LENGTH_LONG).show();
        }
        // Setup UI components
        lpChoosePreInstalledModel =
                (ListPreference) findPreference(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY));
        String[] preInstalledModelNames = new String[preInstalledModelDirs.size()];
        for (int i = 0; i < preInstalledModelDirs.size(); i++) {
            preInstalledModelNames[i] = preInstalledModelDirs.get(i).substring(preInstalledModelDirs.get(i).lastIndexOf("/") + 1);
        }
        lpChoosePreInstalledModel.setEntries(preInstalledModelNames);
        lpChoosePreInstalledModel.setEntryValues(preInstalledModelDirs.toArray(new String[preInstalledModelDirs.size()]));
        lpCPUThreadNum = (ListPreference) findPreference(getString(R.string.CPU_THREAD_NUM_KEY));
        lpCPUPowerMode = (ListPreference) findPreference(getString(R.string.CPU_POWER_MODE_KEY));
        etModelDir = (EditTextPreference) findPreference(getString(R.string.MODEL_DIR_KEY));
        etModelDir.setTitle("Model dir (SDCard: " + Utils.getSDCardDirectory() + ")");
        etLabelPath = (EditTextPreference) findPreference(getString(R.string.LABEL_PATH_KEY));
        etLabelPath.setTitle("Label path (SDCard: " + Utils.getSDCardDirectory() + ")");
        etInputWidth = (EditTextPreference) findPreference(getString(R.string.INPUT_WIDTH_KEY));
        etInputHeight = (EditTextPreference) findPreference(getString(R.string.INPUT_HEIGHT_KEY));
        etInputMean = (EditTextPreference) findPreference(getString(R.string.INPUT_MEAN_KEY));
        etInputStd = (EditTextPreference) findPreference(getString(R.string.INPUT_STD_KEY));
        etScoreThreshold = (EditTextPreference) findPreference(getString(R.string.SCORE_THRESHOLD_KEY));
    }

    private void reloadSettingsAndUpdateUI() {
        SharedPreferences sharedPreferences = getPreferenceScreen().getSharedPreferences();

        String selected_model_dir = sharedPreferences.getString(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY),
                getString(R.string.MODEL_DIR_DEFAULT));
        int selected_model_idx = lpChoosePreInstalledModel.findIndexOfValue(selected_model_dir);
        if (selected_model_idx >= 0 && selected_model_idx < preInstalledModelDirs.size() && selected_model_idx != selectedModelIdx) {
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString(getString(R.string.MODEL_DIR_KEY), preInstalledModelDirs.get(selected_model_idx));
            editor.putString(getString(R.string.LABEL_PATH_KEY), preInstalledLabelPaths.get(selected_model_idx));
            editor.putString(getString(R.string.CPU_THREAD_NUM_KEY), preInstalledCPUThreadNums.get(selected_model_idx));
            editor.putString(getString(R.string.CPU_POWER_MODE_KEY), preInstalledCPUPowerModes.get(selected_model_idx));
            editor.putString(getString(R.string.INPUT_WIDTH_KEY), preInstalledInputWidths.get(selected_model_idx));
            editor.putString(getString(R.string.INPUT_HEIGHT_KEY), preInstalledInputHeights.get(selected_model_idx));
            editor.putString(getString(R.string.INPUT_MEAN_KEY), preInstalledInputMeans.get(selected_model_idx));
            editor.putString(getString(R.string.INPUT_STD_KEY), preInstalledInputStds.get(selected_model_idx));
            editor.putString(getString(R.string.SCORE_THRESHOLD_KEY), preInstalledScoreThresholds.get(selected_model_idx));
            editor.commit();
            lpChoosePreInstalledModel.setSummary(selected_model_dir);
            selectedModelIdx = selected_model_idx;
        }

        String model_dir = sharedPreferences.getString(getString(R.string.MODEL_DIR_KEY),
                getString(R.string.MODEL_DIR_DEFAULT));
        String label_path = sharedPreferences.getString(getString(R.string.LABEL_PATH_KEY),
                getString(R.string.LABEL_PATH_DEFAULT));
        String cpu_thread_num = sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT));
        String cpu_power_mode = sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                getString(R.string.CPU_POWER_MODE_DEFAULT));
        String input_width = sharedPreferences.getString(getString(R.string.INPUT_WIDTH_KEY),
                getString(R.string.INPUT_WIDTH_DEFAULT));
        String input_height = sharedPreferences.getString(getString(R.string.INPUT_HEIGHT_KEY),
                getString(R.string.INPUT_HEIGHT_DEFAULT));
        String input_mean = sharedPreferences.getString(getString(R.string.INPUT_MEAN_KEY),
                getString(R.string.INPUT_MEAN_DEFAULT));
        String input_std = sharedPreferences.getString(getString(R.string.INPUT_STD_KEY),
                getString(R.string.INPUT_STD_DEFAULT));
        String score_threshold = sharedPreferences.getString(getString(R.string.SCORE_THRESHOLD_KEY),
                getString(R.string.SCORE_THRESHOLD_DEFAULT));

        etModelDir.setSummary(model_dir);
        etLabelPath.setSummary(label_path);
        lpCPUThreadNum.setValue(cpu_thread_num);
        lpCPUThreadNum.setSummary(cpu_thread_num);
        lpCPUPowerMode.setValue(cpu_power_mode);
        lpCPUPowerMode.setSummary(cpu_power_mode);
        etInputWidth.setSummary(input_width);
        etInputWidth.setText(input_width);
        etInputHeight.setSummary(input_height);
        etInputHeight.setText(input_height);
        etInputMean.setSummary(input_mean);
        etInputMean.setText(input_mean);
        etInputStd.setSummary(input_std);
        etInputStd.setText(input_std);
        etScoreThreshold.setSummary(score_threshold);
        etScoreThreshold.setText(score_threshold);
    }

    static boolean checkAndUpdateSettings(Context ctx) {
        boolean settingsChanged = false;
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(ctx);

        String model_dir = sharedPreferences.getString(ctx.getString(R.string.MODEL_DIR_KEY),
                ctx.getString(R.string.MODEL_DIR_DEFAULT));
        settingsChanged |= !modelDir.equalsIgnoreCase(model_dir);
        modelDir = model_dir;

        String label_path = sharedPreferences.getString(ctx.getString(R.string.LABEL_PATH_KEY),
                ctx.getString(R.string.LABEL_PATH_DEFAULT));
        settingsChanged |= !labelPath.equalsIgnoreCase(label_path);
        labelPath = label_path;

        String cpu_thread_num = sharedPreferences.getString(ctx.getString(R.string.CPU_THREAD_NUM_KEY),
                ctx.getString(R.string.CPU_THREAD_NUM_DEFAULT));
        settingsChanged |= cpuThreadNum != Integer.parseInt(cpu_thread_num);
        cpuThreadNum = Integer.parseInt(cpu_thread_num);

        String cpu_power_mode = sharedPreferences.getString(ctx.getString(R.string.CPU_POWER_MODE_KEY),
                ctx.getString(R.string.CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpuPowerMode.equalsIgnoreCase(cpu_power_mode);
        cpuPowerMode = cpu_power_mode;

        String input_width = sharedPreferences.getString(ctx.getString(R.string.INPUT_WIDTH_KEY),
                ctx.getString(R.string.INPUT_WIDTH_DEFAULT));
        settingsChanged |= inputWidth != Integer.parseInt(input_width);
        inputWidth = Integer.parseInt(input_width);

        String input_height = sharedPreferences.getString(ctx.getString(R.string.INPUT_HEIGHT_KEY),
                ctx.getString(R.string.INPUT_HEIGHT_DEFAULT));
        settingsChanged |= inputHeight != Integer.parseInt(input_height);
        inputHeight = Integer.parseInt(input_height);

        String input_mean = sharedPreferences.getString(ctx.getString(R.string.INPUT_MEAN_KEY),
                ctx.getString(R.string.INPUT_MEAN_DEFAULT));
        float[] array_data = Utils.parseFloatsFromString(input_mean, ",");
        settingsChanged |= array_data.length != inputMean.length;
        if (!settingsChanged) {
            for (int i = 0; i < array_data.length; i++) {
                settingsChanged |= array_data[i] != inputMean[i];
            }
        }
        inputMean = array_data;

        String input_std = sharedPreferences.getString(ctx.getString(R.string.INPUT_STD_KEY),
                ctx.getString(R.string.INPUT_STD_DEFAULT));
        array_data = Utils.parseFloatsFromString(input_std, ",");
        settingsChanged |= array_data.length != inputStd.length;
        if (!settingsChanged) {
            for (int i = 0; i < array_data.length; i++) {
                settingsChanged |= array_data[i] != inputStd[i];
            }
        }
        inputStd = array_data;

        String score_threshold = sharedPreferences.getString(ctx.getString(R.string.SCORE_THRESHOLD_KEY),
                ctx.getString(R.string.SCORE_THRESHOLD_DEFAULT));
        settingsChanged |= scoreThreshold != Float.parseFloat(score_threshold);
        scoreThreshold = Float.parseFloat(score_threshold);

        return settingsChanged;
    }

    static void resetSettings() {
        selectedModelIdx = -1;
        modelDir = "";
        labelPath = "";
        cpuThreadNum = 0;
        cpuPowerMode = "";
        inputWidth = 0;
        inputHeight = 0;
        inputMean = new float[]{};
        inputStd = new float[]{};
        scoreThreshold = 0;
    }

    @Override
    protected void onResume() {
        super.onResume();
        getPreferenceScreen().getSharedPreferences().registerOnSharedPreferenceChangeListener(this);
        reloadSettingsAndUpdateUI();
    }

    @Override
    protected void onPause() {
        super.onPause();
        getPreferenceScreen().getSharedPreferences().unregisterOnSharedPreferenceChangeListener(this);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        reloadSettingsAndUpdateUI();
    }
}
