package src

type MindletConfig struct {
	PythonRoot      string
	ProjectRoot     string
	ChatModelDir    string
	ModelManagerDir string
	LogDir          string
	StreamLogs      bool
}

func DefaultMindletConfig() *MindletConfig {
	return &MindletConfig{
		PythonRoot:      "/home/nlpfollower/anaconda3/envs/llama-3/bin/python",
		ProjectRoot:     "/home/nlpfollower/Desktop/Web",
		ChatModelDir:    "chat_model",
		ModelManagerDir: "model_manager",
		LogDir:          "",
	}
}
