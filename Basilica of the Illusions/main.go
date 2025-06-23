package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/budgies-nest/budgie/agents"
	"github.com/budgies-nest/budgie/enums/base"
	"github.com/budgies-nest/budgie/helpers"
	"github.com/budgies-nest/budgie/rag"
	"github.com/openai/openai-go"
)

func main() {
	embeddingsAgent, err := agents.NewAgent("Room Keeper",
		agents.WithDMR(context.Background(), base.DockerModelRunnerContainerURL),
		agents.WithEmbeddingParams(
			openai.EmbeddingNewParams{
				Model: "ai/mxbai-embed-large",
			},
		),
		agents.WithMemoryVectorStore("embeddings.json"),
	)
	if err != nil {
		panic(err)
	}

	err = loadOrCreateVectorStore(embeddingsAgent)
	if err != nil {
		panic(err)
	}

	dungeonMaster, err := agents.NewAgent("Bob the Dungeon Master",
		agents.WithDMR(context.Background(), base.DockerModelRunnerContainerURL),
		agents.WithParams(openai.ChatCompletionNewParams{
			//Model:       "ai/qwen2.5:latest",
			//Temperature: openai.Opt(1.0),
			Model:       "ai/qwen2.5:3B-F16",
			//Model:       "ai/qwen2.5:1.5B-F16",
			Temperature: openai.Opt(0.0),
			TopP: 	  openai.Opt(0.1),
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage(`Vous vous appelez Bob, vous êtes le Maître de Donjon, 
				vous êtes un assistant utile qui aide l'utilisateur à explorer la Basilique des Illusions. 
				Vous pouvez répondre aux questions sur la Basilique et son histoire, 
				et vous pouvez aussi aider l'utilisateur à s'orienter. 
				Utilisez uniquement la base de connaissances pour répondre aux questions.
				Concentrez-vous sur la Basilique des Illusions et evitex les répetitions inutiles.
				`),
			},
		}),
	)
	if err != nil {
		panic(err)
	}

	userQuestion := "Parle moi de la Basilique des Illusions et donne moi la liste des différentes zones de la basilique.."

	similarities, err := embeddingsAgent.RAGMemorySearchSimilaritiesWithText(userQuestion, 0.5)

	if err != nil {
		panic(err)
	}
	if len(similarities) == 0 {
		fmt.Println("Aucune information pertinente trouvée dans la base de connaissances.")
		return
	}
	fmt.Println("Informations pertinentes trouvées dans la base de connaissances :")
	for _, similarity := range similarities {
		fmt.Println("-", similarity)
		fmt.Println(strings.Repeat("-", 50))
	}

	dungeonMaster.Params.Messages = append(dungeonMaster.Params.Messages,
		openai.SystemMessage("BASE DE CONNAISSANCES:\n\n"+strings.Join(similarities, "\n\n")),
		openai.UserMessage(userQuestion),
	)

	_, err = dungeonMaster.ChatCompletionStream(func(self *agents.Agent, content string, err error) error {
		fmt.Print(content)
		return nil
	})
	if err != nil {
		panic(err)
	}

}

func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}

func loadOrCreateVectorStore(embeddingsAgent *agents.Agent) error {
	if !fileExists("embeddings.json") {
		// Create a new memory vector store and embedding file
		embeddingsAgent.ResetMemoryVectorStore()
		markdownContent, err := helpers.ReadTextFile("README.md")
		if err != nil {
			return err
		}
		chunks := rag.ChunkWithMarkdownHierarchy(markdownContent)
		_, err = embeddingsAgent.CreateAndSaveEmbeddingFromChunks(chunks)
		if err != nil {
			return err
		}
		embeddingsAgent.PersistMemoryVectorStore()

		// NOTE: add a method: GenerateEmbeddingsFromMarkdownHierarchy

	} else {
		err := embeddingsAgent.LoadMemoryVectorStore()
		if err != nil {
			return err
		}
	}
	return nil
}
